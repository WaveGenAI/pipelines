import csv


def append_lyrics_to_csv_batch(updates: list, csv_path: str):
    """
    Update the corresponding rows in the CSV to add lyrics for given audio files in batch.

    Parameters:
    - updates: A list of tuples where each tuple contains (audio_name, lyrics).
    - csv_path: The path to the CSV file (string).
    """
    # Read the entire CSV file into memory
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)

    header = rows[0]
    if "lyrics" not in header:
        header.append("lyrics")
        lyrics_index = len(header) - 1
    else:
        lyrics_index = header.index("lyrics")

    updates_dict = {audio_name: lyrics for audio_name, lyrics in updates}

    # Update the rows with lyrics from the updates list
    for row in rows[1:]:  # Skip the header
        audio_name = row[0]
        if audio_name in updates_dict:
            lyrics = updates_dict[audio_name]
            if len(row) > lyrics_index:
                row[lyrics_index] = lyrics
            else:
                row.append(lyrics)

    # Write the updated rows back to the CSV file
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
