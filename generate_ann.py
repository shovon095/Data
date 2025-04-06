import os
import argparse

def read_predictions(prediction_file_path):
    """
    Read token-label pairs grouped by RecordID (without 'RecordID' prefix)
    """
    records = {}
    current_record_id = None

    with open(prediction_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            token, label = parts

            if token.startswith("RecordID"):
                current_record_id = token.replace("RecordID", "")
                records[current_record_id] = []
                continue

            if current_record_id is not None:
                records[current_record_id].append((token, label))

    return records

def create_annotations(records, data_doc_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(data_doc_dir):
        if not file.endswith(".txt"):
            continue
        record_id = file.replace(".txt", "")
        output_file = os.path.join(output_dir, f"{record_id}.ann")
        record_file = os.path.join(data_doc_dir, file)

        annotations = []
        term_index = 1

        if record_id in records:
            with open(record_file, "r") as rf:
                record_text = rf.read()

            token_label_pairs = records[record_id]
            token_end = 0
            i = 0
            while i < len(token_label_pairs):
                token, label = token_label_pairs[i]

                if label.startswith("B"):
                    entity = token
                    entity_type = label.split("-")[-1]
                    token_start = record_text.find(token, token_end)
                    token_end = token_start + len(token)

                    i += 1
                    while i < len(token_label_pairs) and token_label_pairs[i][1].startswith("I"):
                        next_token = token_label_pairs[i][0]
                        entity += " " + next_token
                        next_start = record_text.find(next_token, token_end)
                        token_end = next_start + len(next_token)
                        i += 1

                    annotations.append(f"T{term_index}\t{entity_type} {token_start} {token_end}\t{entity}")
                    annotations.append(f"E{term_index}\t{entity_type}:T{term_index}")
                    term_index += 1
                else:
                    i += 1
        else:
            print(f"\u26a0\ufe0f No predictions found for {record_id}. Writing empty ann.")

        with open(output_file, "w") as of:
            for ann in annotations:
                of.write(ann + "\n")

        print(f"✅ Annotations written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate .ann annotation files from predictions.")
    parser.add_argument("prediction_file_path", help="Path to the prediction .txt file")
    parser.add_argument("data_doc_dir", help="Directory containing .txt documents for each record")
    parser.add_argument("output_dir", help="Directory to write .ann output files")

    args = parser.parse_args()

    records = read_predictions(args.prediction_file_path)
    create_annotations(records, args.data_doc_dir, args.output_dir)

if __name__ == "__main__":
    main()
