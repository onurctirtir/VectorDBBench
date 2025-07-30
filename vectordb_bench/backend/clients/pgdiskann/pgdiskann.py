"""Wrapper around the pg_diskann vector database over VectorDB"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from ..api import VectorDB
from .config import PgDiskANNConfigDict, PgDiskANNIndexConfig

log = logging.getLogger(__name__)


class PgDiskANN(VectorDB):
    """Use psycopg instructions"""

    conn: psycopg.Connection[Any] | None = None
    coursor: psycopg.Cursor[Any] | None = None

    _filtered_search: sql.Composed
    _unfiltered_search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: PgDiskANNConfigDict,
        db_case_config: PgDiskANNIndexConfig,
        collection_name: str = "pg_diskann_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "PgDiskANN"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "pgdiskann_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        self.conn, self.cursor = self._create_connection(**self.db_config)

        log.info(f"{self.name} config values: {self.db_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {self.db_config}\n{self.case_config}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        cursor = conn.cursor()
        
        # Enable extensions - Azure Database for PostgreSQL has pre-installed extensions
        # We need to enable them, not create them
        try:
            # Check which extensions are already enabled
            cursor.execute("SELECT extname FROM pg_extension WHERE extname IN ('pg_diskann', 'vector', 'citus');")
            enabled_extensions = [row[0] for row in cursor.fetchall()]
            
            # Enable pg_diskann extension if not already enabled
            if 'pg_diskann' not in enabled_extensions:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_diskann CASCADE")
                log.info("PgDiskANN extension enabled successfully")
            
            # Enable vector extension if not already enabled
            if 'vector' not in enabled_extensions:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector CASCADE")
                log.info("Vector extension enabled successfully")
                
            # Enable citus extension if not already enabled (for Azure Citus)
            if 'citus' not in enabled_extensions:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS citus CASCADE")
                log.info("Citus extension enabled successfully")
                
            conn.commit()
            
        except Exception as e:
            log.warning(f"Extension setup warning: {e}")
            # Try to continue anyway, extensions might already be available
            conn.rollback()
        
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # index configuration may have commands defined that we should set during each client session
        session_options: dict[str, Any] = self.case_config.session_param()

        if len(session_options) > 0:
            for setting_name, setting_val in session_options.items():
                command = sql.SQL("SET {setting_name} " + "= {setting_val};").format(
                    setting_name=sql.Identifier(setting_name),
                    setting_val=sql.Identifier(str(setting_val)),
                )
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            self.conn.commit()

        self._filtered_search = sql.Composed(
            [
                sql.SQL(
                    "SELECT id FROM public.{table_name} WHERE id >= %s ORDER BY embedding ",
                ).format(table_name=sql.Identifier(self.table_name)),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(" %s::vector LIMIT %s::int"),
            ],
        )

        self._unfiltered_search = sql.Composed(
            [
                sql.SQL("SELECT id FROM public.{} ORDER BY embedding ").format(
                    sql.Identifier(self.table_name),
                ),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(" %s::vector LIMIT %s::int"),
            ],
        )

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        self._post_insert()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {index_name}").format(
            index_name=sql.Identifier(self._index_name),
        )
        log.debug(drop_index_sql.as_string(self.cursor))
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _set_parallel_index_build_param(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        if index_param["maintenance_work_mem"] is not None:
            self.cursor.execute(
                sql.SQL("SET maintenance_work_mem TO {};").format(
                    index_param["maintenance_work_mem"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET maintenance_work_mem TO {};").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["maintenance_work_mem"],
                ),
            )
            self.conn.commit()

        if index_param["max_parallel_workers"] is not None:
            self.cursor.execute(
                sql.SQL("SET max_parallel_maintenance_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_maintenance_workers TO '{}';").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("SET max_parallel_workers TO '{}';").format(
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER USER {} SET max_parallel_workers TO '{}';").format(
                    sql.Identifier(self.db_config["user"]),
                    index_param["max_parallel_workers"],
                ),
            )
            self.cursor.execute(
                sql.SQL("ALTER TABLE {} SET (parallel_workers = {});").format(
                    sql.Identifier(self.table_name),
                    index_param["max_parallel_workers"],
                ),
            )
            self.conn.commit()

        results = self.cursor.execute(sql.SQL("SHOW max_parallel_maintenance_workers;")).fetchall()
        results.extend(self.cursor.execute(sql.SQL("SHOW max_parallel_workers;")).fetchall())
        results.extend(self.cursor.execute(sql.SQL("SHOW maintenance_work_mem;")).fetchall())
        log.info(f"{self.name} parallel index creation parameters: {results}")

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param: dict[str, Any] = self.case_config.index_param()
        self._set_parallel_index_build_param()

        options = []
        for option_name, option_val in index_param["options"].items():
            if option_val is not None:
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option_name),
                        val=sql.Identifier(str(option_val)),
                    ),
                )

        with_clause = sql.SQL("WITH ({});").format(sql.SQL(", ").join(options)) if any(options) else sql.Composed(())

        index_create_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
            USING {index_type} (embedding {embedding_metric})
            """,
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            index_type=sql.Identifier(index_param["index_type"].lower()),
            embedding_metric=sql.Identifier(index_param["metric"]),
        )
        index_create_sql_with_with_clause = (index_create_sql + with_clause).join(" ")
        log.debug(index_create_sql_with_with_clause.as_string(self.cursor))
        self.cursor.execute(index_create_sql_with_with_clause)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            # Create the table first
            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{table_name} (id BIGINT PRIMARY KEY, embedding vector({dim}));",
                ).format(table_name=sql.Identifier(self.table_name), dim=dim),
            )
            self.conn.commit()

            # Check if Citus extension is available and create distributed table
            self._create_distributed_table()
            
        except Exception as e:
            log.warning(f"Failed to create pgdiskann table: {self.table_name} error: {e}")
            raise e from None

    def _create_distributed_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        # Check if Citus distribution is enabled in config
        if not getattr(self.case_config, 'enable_citus_distribution', True):
            log.info(f"{self.name} Citus distribution disabled in config")
            return

        try:
            # Check if Citus extension is enabled (Azure Citus should have it pre-installed)
            self.cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'citus');"
            )
            citus_available = self.cursor.fetchone()[0]
            
            if citus_available:
                log.info(f"{self.name} Citus extension found, creating distributed table")
                
                # Set shard count from config BEFORE creating distributed table (optimized for 500K rows)
                shard_count = getattr(self.case_config, 'shard_count', 8)
                log.info(f"{self.name} Setting shard count to {shard_count}")
                
                # Apply shard count setting
                self.cursor.execute(f"SET citus.shard_count = {shard_count}")
                self.conn.commit()
                
                # Create distributed table with hash distribution on id column
                self.cursor.execute(
                    sql.SQL("SELECT create_distributed_table({table_name}, 'id');").format(
                        table_name=sql.Literal(self.table_name)
                    )
                )
                self.conn.commit()
                
                # Verify shard creation with single query
                self.cursor.execute(
                    sql.SQL("SELECT count(*) FROM pg_dist_shard WHERE logicalrelid = {table_name}::regclass;").format(
                        table_name=sql.Literal(self.table_name)
                    )
                )
                actual_shard_count = self.cursor.fetchone()[0]
                log.info(f"{self.name} Successfully created distributed table with {actual_shard_count} shards")
                
            else:
                log.warning(f"{self.name} Citus extension not found, creating regular table")
                
        except Exception as e:
            log.warning(f"Failed to create distributed table: {e}")
            log.info(f"{self.name} Falling back to regular table creation")

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                ),
            ) as copy:
                copy.set_types(["bigint", "vector"])
                for i, row in enumerate(metadata_arr):
                    copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            if kwargs.get("last_batch"):
                self._post_insert()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        q = np.asarray(query)
        if filters:
            gt = filters.get("id")
            result = self.cursor.execute(
                self._filtered_search,
                (gt, q, k),
                prepare=True,
                binary=True,
            )
        else:
            result = self.cursor.execute(self._unfiltered_search, (q, k), prepare=True, binary=True)

        return [int(i[0]) for i in result.fetchall()]

    def collect_post_benchmark_config(self) -> dict:
        """
        Collect comprehensive database configuration metrics after benchmark completion.
        This runs while data is still loaded to capture actual runtime state.
                
        Returns:
            dict: Comprehensive configuration metrics collected from the live database
        """
        try:
            # Re-establish connection for post-benchmark analysis if needed
            with psycopg.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    
                    log.info("🔍 POST_BENCHMARK_ANALYSIS_START")
                    
                    config_metrics = {
                        'collection_timestamp': datetime.now().isoformat(),
                        'data_still_loaded': True,
                        'analysis_phase': 'post_benchmark'
                    }
                    
                    
                    # 1. Current GUC parameters (live state) - ACTIVE
                    config_metrics['current_guc_parameters'] = self._collect_live_guc_parameters(cursor)
                    
                    # 2. DiskANN parameters (always collected) - ACTIVE
                    config_metrics['diskann_parameters'] = self._collect_diskann_parameters(cursor)
                    
                    # 3. Current Citus state (if enabled) - ACTIVE
                    if self.case_config.enable_citus_distribution:
                        config_metrics['citus_runtime_state'] = self._collect_citus_runtime_state(cursor)
                 
                    log.info("🔍 POST_BENCHMARK_ANALYSIS_END")
                    
                    return config_metrics
                    
        except Exception as e:
            log.error(f"Error collecting post-benchmark config: {e}")
            return {'error': str(e)}
    
    
    def _collect_live_guc_parameters(self, cursor) -> dict:
        """Collect current GUC parameter values in live system."""
        try:
            guc_params = {}
            
            # Define only the essential Citus parameters to collect
            citus_parameters = [
                'citus.max_adaptive_executor_pool_size',
                'citus.max_cached_connection_lifetime',
                'citus.max_cached_conns_per_worker',
                'citus.max_client_connections',
                'citus.max_shared_pool_size',
                'citus.local_shared_pool_size',
                'citus.executor_slow_start_interval',
                'citus.force_max_query_parallelization',
                'citus.enable_binary_protocol',
                'citus.enable_repartition_joins'
            ]
            
            # Use simple SHOW commands for each parameter
            successful_params = 0
            failed_params = 0
            
            for param in citus_parameters:
                try:
                    cursor.execute(f"SHOW {param}")
                    result = cursor.fetchone()
                    guc_params[param] = result[0] if result else 'NOT_AVAILABLE'
                    successful_params += 1
                except Exception as e:
                    guc_params[param] = f'ERROR: {str(e)}'
                    failed_params += 1
            
            # Add collection metadata
            guc_params['_collection_info'] = {
                'timestamp': datetime.now().isoformat(),
                'total_parameters_requested': len(citus_parameters),
                'successful_parameters': successful_params,
                'failed_parameters': failed_params,
                'method': 'SHOW_commands',
                'success_rate': f"{(successful_params/len(citus_parameters)*100):.1f}%"
            }
            
            log.info(f"Collected {successful_params}/{len(citus_parameters)} Citus GUC parameters using SHOW commands")
            return guc_params
            
        except Exception as e:
            log.error(f"Error collecting Citus GUC parameters: {e}")
            return {'error': str(e)}
    
    def _collect_diskann_parameters(self, cursor) -> dict:
        """Collect DiskANN GUC parameters using SHOW commands."""
        try:
            diskann_params = {}
            
            # First, set session parameters exactly like in init() method
            try:
                session_options: dict[str, Any] = self.case_config.session_param()
                
                if len(session_options) > 0:
                    for setting_name, setting_val in session_options.items():
                        if 'diskann' in setting_name.lower():  # Only set DiskANN params
                            command = sql.SQL("SET {setting_name} = {setting_val};").format(
                                setting_name=sql.Identifier(setting_name),
                                setting_val=sql.Literal(str(setting_val)),
                            )
                            cursor.execute(command)
                            log.info(f"Set DiskANN parameter: {setting_name} = {setting_val}")
                
                # Also set iterative_search to a default value since it's not in session_param()
                try:
                    cursor.execute("SET diskann.iterative_search = 'Relaxed_Order'")
                    log.info("Set DiskANN parameter: diskann.iterative_search = Relaxed_Order (default)")
                except Exception as e:
                    log.warning(f"Failed to set diskann.iterative_search: {e}")
                
                cursor.connection.commit()
                log.info("Successfully applied DiskANN session parameters")
                
            except Exception as e:
                log.warning(f"Failed to set DiskANN session parameters: {e}")
            
            # Define the 2 essential DiskANN parameters to collect
            diskann_guc_parameters = [
                'diskann.l_value_is',
                'diskann.iterative_search'
            ]
            
            # Use simple SHOW commands for each parameter
            successful_params = 0
            failed_params = 0
            
            for param in diskann_guc_parameters:
                try:
                    cursor.execute(f"SHOW {param}")
                    result = cursor.fetchone()
                    diskann_params[param] = result[0] if result else 'NOT_AVAILABLE'
                    successful_params += 1
                except Exception as e:
                    diskann_params[param] = f'ERROR: {str(e)}'
                    failed_params += 1
            
            # Add collection metadata
            diskann_params['_collection_info'] = {
                'timestamp': datetime.now().isoformat(),
                'total_parameters_requested': len(diskann_guc_parameters),
                'successful_parameters': successful_params,
                'failed_parameters': failed_params,
                'method': 'SHOW_commands_with_session_setup',
                'success_rate': f"{(successful_params/len(diskann_guc_parameters)*100):.1f}%"
            }
            
            log.info(f"Collected {successful_params}/{len(diskann_guc_parameters)} DiskANN GUC parameters using SHOW commands")
            return diskann_params
            
        except Exception as e:
            log.error(f"Error collecting DiskANN GUC parameters: {e}")
            return {'error': str(e)}
    
    def _collect_citus_runtime_state(self, cursor) -> dict:
        """Collect current Citus runtime state."""
        try:
            # Get shard distribution across workers - single query only
            cursor.execute("""
                SELECT nodename, nodeport, count(*) as shard_count_on_node
                FROM pg_dist_shard
                JOIN pg_dist_placement USING (shardid)
                JOIN pg_class ON (logicalrelid = oid)
                JOIN pg_dist_node USING (groupid)
                WHERE relname = 'pg_diskann_collection'
                GROUP BY (nodename, nodeport)
                ORDER BY nodeport
            """)
            results = cursor.fetchall()
            
            return {
                'shards_per_worker': [
                    {
                        'worker_node': row[0],
                        'worker_port': row[1], 
                        'shard_count': row[2]
                    } for row in results
                ]
            }
            
        except Exception as e:
            log.error(f"Error collecting Citus runtime state: {e}")
            return {'error': str(e)}
    