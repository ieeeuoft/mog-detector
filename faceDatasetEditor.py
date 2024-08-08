import csv
import os

labelDict = {
    "angry": 0,
    "fear": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5
}

countDict = {
    "happy": 0,
    "neither": 0,
    "sad": 0,
    "mog": 0
}


def main():
    input_file = "./data/face/processed_landmarks_2.csv"
    output_file = "./data/face/expanded_landmarks.csv"

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
# , open(output_file, mode='w', newline='') as outfile:
    with open(input_file, mode='r') as infile:
        reader = csv.reader(infile)
        # writer = csv.writer(outfile)

        # header = []
        # for x in range(468):
        #     header.extend([f'Landmark_{x}_x', f'Landmark_{x}_y', f'Landmark_{x}_z'])
        # header.append('label')

        # Read the header
        next(reader)
        # writer.writerow(header)

        for row in reader:
            assert len(row) == 53, f"Invalid header length: {len(row)}"
            label = row[0]
            countDict[label] += 1

    #     for row in reader:
    #         label = row[0]
    #         if label == "disgust":
    #             continue
    #         if countDict[label] == 3000:
    #             continue
    #         new_row = []
    #         for landmark in row[1:]:
    #             x, y, z = landmark[1:-1].split(',')
    #             new_row.extend([x, y, z])
    #         new_row.append(labelDict[label])
    #         countDict[label] += 1
    #         writer.writerow(new_row)
    #         assert len(new_row) == 468 * 3 + 1, f"Invalid row length: {len(new_row)}"
    print("Count Dictionary:\n", countDict)
    print("Expanded landmarks saved to", output_file)


main()