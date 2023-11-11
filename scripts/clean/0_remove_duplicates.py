import csv


def remove_duplicates():
    title = None
    data = []
    names = set()

    # `data.csv` is not available in this repo for privacy reasons
    with open('data/data.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        title = next(csv_reader)
        next(csv_reader)
        next(csv_reader)

        # remove duplicates and incomplete entries
        for line in csv_reader:
            name = line[-3].lower()
            if name != "" and name not in names:
                names.add(name)
                tmp = {title[i]: line[i] for i in range(len(line))}
                data.append(tmp)
            else:
                print(name)
    print('Total # participants:', len(data), '\n')

    # remove personally identifiable information
    title = title[:-3]

    # `data_anonymized.csv` is also not available for privacy reasons
    with open('data/data_anonymized.csv', 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(title)
        for d in data:
            line = [d[x] for x in title if x in d]
            csv_writer.writerow(line)


if __name__ == '__main__':
    remove_duplicates()