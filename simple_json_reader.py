#!/usr/bin/env python3
"""
Simple JSON to Readable Text Converter
Clean, simple output with better readability
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path


def convert_json_to_readable(input_file, output_dir="analyses"):
    """Convert JSON file to readable text format."""
    # Load JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    input_name = Path(input_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{input_name}_readable_{timestamp}.txt"
    
    # Write readable output
    with open(output_file, 'w') as f:
        f.write(f"Benchmark Result Analysis\n")
        f.write(f"========================\n")
        f.write(f"Source: {input_file}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Key highlights first
        f.write("KEY RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        if 'results' in data and data['results']:
            result = data['results'][0]
            metrics = result.get('metrics', {})
            
            # Performance summary
            f.write(f"Run ID: {data.get('run_id', 'N/A')}\n")
            f.write(f"Load Time: {metrics.get('load_duration', 'N/A')} seconds\n")
            f.write(f"Insert Time: {metrics.get('insert_duration', 'N/A')} seconds\n")
            f.write(f"Index Build Time: {metrics.get('optimize_duration', 'N/A')} seconds\n")
            f.write(f"Max QPS: {max(metrics.get('conc_qps_list', [0])) if metrics.get('conc_qps_list') else 'N/A'}\n")
            f.write(f"Recall: {metrics.get('recall', 'N/A')}\n")
            f.write(f"NDCG: {metrics.get('ndcg', 'N/A')}\n")
            
            # Configuration summary
            if 'task_config' in data:
                config = data['task_config']
                db_config = config.get('db_case_config', {})
                f.write(f"\nDatabase: {config.get('db', 'N/A')}\n")
                f.write(f"Index Type: {db_config.get('index', 'N/A')}\n")
                f.write(f"Metric: {db_config.get('metric_type', 'N/A')}\n")
                f.write(f"Max Neighbors: {db_config.get('max_neighbors', 'N/A')}\n")
                f.write(f"L Value (build): {db_config.get('l_value_ib', 'N/A')}\n")
                f.write(f"L Value (search): {db_config.get('l_value_is', 'N/A')}\n")
                f.write(f"Citus Enabled: {db_config.get('enable_citus_distribution', 'N/A')}\n")
                f.write(f"Shard Count: {db_config.get('shard_count', 'N/A')}\n")
        
        f.write("\n\n")
        
        # Full data dump with better structure
        f.write("COMPLETE DATA:\n")
        f.write("=" * 60 + "\n")
        f.write(json.dumps(data, indent=2, sort_keys=True))
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Convert JSON result files to readable text')
    parser.add_argument('input', help='Input JSON file or directory')
    parser.add_argument('-o', '--output', default='analyses', help='Output directory')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Single file
        print(f"Converting: {args.input}")
        output_file = convert_json_to_readable(args.input, args.output)
        print(f"✅ Created: {output_file}")
        
    elif os.path.isdir(args.input):
        # Directory
        json_files = list(Path(args.input).glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            print(f"Converting: {json_file.name}")
            try:
                output_file = convert_json_to_readable(json_file, args.output)
                print(f"  ✅ Created: {output_file.name}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
    else:
        print(f"❌ Not found: {args.input}")


if __name__ == "__main__":
    main()
