"""
Convert the training data format from XML to BIO format to train a tagger.

python3 convert_xml_to_bio.py \
    --xml_file ../../data/experimental/time_data_eng.xml \
    --output_file ../../data/experimental/time_data_eng.csv 
"""

import argparse
import re
import pandas as pd
import numpy as np

def parse_line(line):

    bio_line = []
    values = []

    line = line.replace('\n', '')
    if line.strip() == '':
        return None, None
    if line.strip()[0] == '#':
        return None, None
    
    annotation_types = ["TIMEX3", "LOCATION"]
    annotations = []

    for annotation_type in annotation_types:
        pattern = rf'<{annotation_type}\b[^>]*?>.*?<\/{annotation_type}>'
        for match in re.finditer(pattern, line):
            if annotation_type == "TIMEX3":
                # Extract the type attribute from TIMEX3 tags
                type_match = re.search(r'type="([^"]+)"', match.group())
                if type_match:
                    annotation_type = type_match.group(1)
                value = re.search(r'value="([^"]+)"', match.group())
                start_index = value.start()
                end_index = value.end()
                true_value = value.group(0)[len('value="'):-1]
            else:
                true_value = re.sub(r"</?[^>]+>", "", match.group())
                start_index = match.start()
                end_index = match.end()
            annotations.append((match, annotation_type, start_index, end_index, true_value))

    # Convert the line to BIO format
    position = 0
    bio_line = []

    current_annotation = None
    current_annotation_type = None

    values = []
    while position < len(line):
        next_whitespace = line.find(' ', position)
        if next_whitespace == -1:
            next_whitespace = len(line) # No more whitespace, take the rest of the line 
        next_span = line[position:next_whitespace]
        
        # detect if we are in an annotation
        in_annotation = False
        for annotation, annotation_type, start_index, end_index, true_value in annotations:
            if position >= annotation.start() and position < annotation.end():
                in_annotation = True
                break
        
        if in_annotation:
            next_span = annotation.group()
            next_span = re.findall(r'\>.*\<', next_span)[0]
            next_span = next_span[1:-1]
            
            tokens = next_span.split(" ")
            for i in range(len(tokens)):
                token = tokens[i]
                if i == 0:
                    bio_line.append((token, f'B-{annotation_type}'))
                else:
                    bio_line.append((token, f'I-{annotation_type}'))

            position = annotation.end()

            values.append({
                "start_index": start_index,
                "end_index": end_index,
                "true_value": true_value,
                "surface_value": next_span,
                "annotation_type": annotation_type}
            )
        else:
            next_token = next_span
            position = next_whitespace + 1
            bio_line.append((next_token, 'O'))

    return bio_line, values

if __name__ == "__main__":

    np.random.seed(0) 
    
    parser = argparse.ArgumentParser(description="Convert XML file to BIO format.")
    parser.add_argument('--xml_file', type=str, help='Path to the input XML file.')
    parser.add_argument('--output_file', type=str, help='Path to the output BIO file.')
    parser.add_argument('--print', action="store_true", default=False, help='Print outputs to console')
    args = parser.parse_args()

    i_sample = 0
    output = []

    for line in open(args.xml_file, 'r'):
        bio_line, values = parse_line(line)
        if bio_line is None:
            continue
        dataset = np.random.choice(("train", "valid", "test"), p=(0.8, 0.1, 0.1))
        for line in bio_line:
            output.append([i_sample, line[0], line[1], dataset])
            if args.print:
                print(f"{line[0]}\t{line[1]}\n", end="")

        if args.print:
            print("\n")

        i_sample += 1

    output = pd.DataFrame(output, columns=["sentence_id", "words", "labels", "dataset"])
    output.to_csv(args.output_file, index=False)
    print("wrote " + args.output_file)
