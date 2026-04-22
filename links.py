import re
import csv
from datetime import datetime

def parse_stream_file(filename):
    """
    Parse the stream list file and extract entries with "Aaron" in the title
    from Jan 1, 2026 to current date (April 21, 2026)
    """
    
    # Define date range
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 4, 21)
    
    # Pattern to match each line
    # Format: Title (Date) - URL
    pattern = r'^(.+?)\s+\((\d{1,2}/\d{1,2}/\d{2,4})\)\s+-\s+(https?://[^\s]+)'
    
    matches = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
                
            match = re.match(pattern, line)
            if match:
                title = match.group(1).strip()
                date_str = match.group(2)
                url = match.group(3)
                
                # Parse the date
                try:
                    # Handle 2-digit and 4-digit years
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        month, day, year = parts
                        year = int(year)
                        if year < 100:
                            year = 2000 + year if year >= 25 else 2000 + year
                        
                        stream_date = datetime(year, int(month), int(day))
                        
                        # Check if date is within range
                        if start_date <= stream_date <= end_date:
                            # Check if "Aaron" is in the title (case-insensitive)
                            if 'aaron' in title.lower():
                                matches.append({
                                    'title': title,
                                    'date': stream_date.strftime('%Y-%m-%d'),
                                    'original_date': date_str,
                                    'url': url,
                                    'line_number': line_num
                                })
                except ValueError as e:
                    print(f"Error parsing date '{date_str}' on line {line_num}: {e}")
                    continue
    
    return matches

def save_to_csv(data, output_file):
    """
    Save the filtered data to a CSV file
    """
    if not data:
        print("No data found matching the criteria.")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'date', 'original_date', 'url', 'line_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"Successfully saved {len(data)} entries to {output_file}")

def main():
    input_file = 'stream_list.txt'
    output_file = 'aaron_streams_2026.csv'
    
    print(f"Reading from: {input_file}")
    print(f"Filtering dates from 2026-01-01 to 2026-04-21")
    print(f"Filtering titles containing 'Aaron'")
    print("-" * 50)
    
    matches = parse_stream_file(input_file)
    
    if matches:
        print(f"Found {len(matches)} matching entries:")
        for match in matches:
            print(f"  - {match['date']}: {match['title'][:60]}...")
        
        save_to_csv(matches, output_file)
        
        # Print summary statistics
        print("\nSummary:")
        print(f"  Total entries with 'Aaron': {len(matches)}")
        
        # Group by date
        date_counts = {}
        for match in matches:
            date = match['date']
            date_counts[date] = date_counts.get(date, 0) + 1
        
        print("\nEntries by date:")
        for date in sorted(date_counts.keys()):
            print(f"  {date}: {date_counts[date]} entries")
            
    else:
        print("No matching entries found.")

if __name__ == "__main__":
    main()