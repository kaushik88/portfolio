import csv


def parse_csv_file(file_name, delimiter=',', quotechar='"', skip_first=False):
    """
    Parse CSV Files and yield 1 row at a time.
    """
    with open(file_name, 'rt') as csvfile:
        filereader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        if skip_first:
            next(filereader)
        for row in filereader:
            yield row


def csv_file_to_dataset(file_name):
    dataset = []
    scene = []
    scene_id = ""
    for row in parse_csv_file(file_name, delimiter='\t', skip_first=True):
        if scene_id and str(row[1]) != scene_id:
            dataset.append({"texts": list(scene), "scene_id": scene_id})
            scene = []
        scene_id = str(row[1])
        scene.append({"speaker_id": str(row[2]), "text": str(row[5])})
    return dataset
