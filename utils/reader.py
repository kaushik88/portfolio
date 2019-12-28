import csv


def parse_csv_file(file_name, delimiter=',', quotechar='"'):
    """
    Parse CSV Files and yield 1 row at a time.
    """
    with open(file_name, 'rt') as csvfile:
        filereader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        for row in filereader:
            yield row
