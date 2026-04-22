#!/usr/bin/env bash

VIDEO_PATHS=(
    "TopstepTV Euro Trade with Deeyana - Live Futures Day Trading .webm"
    "TopstepTV London Open with Aaron Trades Live Futures Day Trading .webm"
    "TopstepTV Market Kickoff with AnneMarie  Live Futures Day Trading .webm"
    "TopstepTV Slow Markets with Dakota  Live Futures Day Trading .webm"
    "TopstepTV At the Buzzer with Coach Jay Live Futures Day Trading Recap.webm"
)

for path in "${VIDEO_PATHS[@]}"; do
    rm "logs.txt"
    rm "errors.txt"
    rm "trades_2_log.csv"
    rm "trades_data.json"
    rm "config.json"

    # Extract filename from full path
    video_name=$(basename "$path")
    echo "Processing: $video_name"

    # Write current video path
    echo "$path" > video_path.txt

    # Reset frame number
    echo "0" > frame_number.txt

    # Write info.json
    cat <<EOF > info.json
{
  "video_name": "$video_name"
}
EOF

    # Run Python with UTF-8 enforced
    python3 -X utf8 run.py 2>&1 | tee -a logs.txt


done
