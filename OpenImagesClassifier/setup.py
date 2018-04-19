""" Setup script for OpenImagesClassifier
    Compose DataSet according to chosen classes.
"""

import sqlite3
import urllib.request

import pandas
import os.path

import multiprocessing
import tqdm
import wget
import tarfile
import shutil
import time

from OpenImagesClassifier import config


def create_tables(conn):
        c = conn.cursor()

        c.execute('''CREATE TABLE Images (
                      ImageID CHAR(16),
                      Subset VARCHAR,
                      OriginalURL VARCHAR,
                      OriginalLandingURL VARCHAR,
                      License VARCHAR,
                      AuthorProfileURL VARCHAR,
                      Author VARCHAR,
                      Title VARCHAR,
                      OriginalSize BIGINT,
                      OriginalMD5 VARCHAR,
                      Thumbnail300KURL VARCHAR,
                      PRIMARY KEY(ImageID)
                    )''')

        c.execute('''CREATE TABLE SelectedImages (
                      ImageID CHAR(16),
                      PRIMARY KEY(ImageID)
                    )''')

        c.execute('''CREATE TABLE Dict (
                      LabelName VARCHAR,
                      DisplayLabelName VARCHAR,
                      ClassNumber INT,
                      PRIMARY KEY (LabelName)
                  )''')

        c.execute('''CREATE TABLE Labels (
                      ImageID CHAR(16) REFERENCES Images(ImageID),
                      Source VARCHAR,
                      LabelName VARCHAR REFERENCES Dict(LabelName),
                      Confidence REAL,
                      PRIMARY KEY(ImageID, Source, LabelName)
                  )''')

        conn.commit()


def fill_table(conn, table_name):
    for i in range(len(config.CSV_FILES[table_name])):
        batch_size = 100000
        for df in pandas.read_csv(((config.CSV_FILES[table_name])[i])['file_path'], chunksize=batch_size,
                                  iterator=True):
            df.to_sql(table_name, conn, if_exists='append', index=False)

        print("Inserted {} into table {}".format(((config.CSV_FILES[table_name])[i])['file_path'], table_name))


def fill_dict(conn, table_name):
    df = pandas.read_csv(((config.CSV_FILES[table_name])[0])['file_path'], names=['LabelName', 'DisplayLabelName'],
                         header=None)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    print("Inserted {} into table {}".format(((config.CSV_FILES[table_name])[0])['file_path'], table_name))


def build_database():
    """Insert downloaded CSV into sqlite database for further selection of categories"""
    if not os.path.exists(config.DATABASE['filename']):
        print("Creates sqlite database for downloaded csv data")
        with sqlite3.connect(config.DATABASE['filename']) as conn:
            create_tables(conn)
            fill_table(conn, 'Images')
            fill_table(conn, 'Labels')
            fill_dict(conn, 'Dict')
            conn.commit()


def download_files():
    """Download .csv files """
    print("Starts file download:")
    for url in config.DOWNLOAD_PATHS:
        print("Downloads tar file from: ", url)
        file_path = wget.download(url)
        mode = "r:gz" if (file_path.endswith("tar.gz")) else "r:"
        tar = tarfile.open(file_path, mode)
        tar.extractall(path=config.DATA_DIRECTORY)
        tar.close()
        os.remove(file_path)


def get_top_n_classes(n):
    """Not used right now, but useful to determine biggest classes"""
    with sqlite3.connect((config.DATABASE['filename'])) as conn:
        c = conn.cursor()
        mode = 'train'
        result = c.execute("""SELECT l.LabelName, COUNT(*)
                      FROM Labels AS l
                      INNER JOIN Images AS img ON l.ImageID = img.ImageID
                      WHERE l.Confidence = 1.0
                      AND img.Subset = ?
                      GROUP BY l.LabelName
                      ORDER BY COUNT(*) DESC
                      LIMIT ?""", (mode, n,))

        print("Selected top {} classes".format(n))
        class_list = []
        for row in result.fetchall():
            class_list.append(row[0])
            label_name = c.execute("""SELECT DisplayLabelName FROM DICT WHERE LabelName = ?""", (row[0],))
            print(label_name.fetchone()[0])

        conn.commit()


def produce_subset():
    with sqlite3.connect(config.DATABASE['filename']) as conn:
        cursor = conn.cursor()
        collect_images(cursor, 'train', config.TRAIN_DATASET['images_per_class'])
        collect_images(cursor, 'test', config.TEST_DATASET['images_per_class'])
        collect_images(cursor, 'validation', config.VALID_DATASET['images_per_class'])
        conn.commit()

        clean_up_database(conn, config.CATEGORIES)


def collect_images(cursor, mode, count):
    for elem in config.CATEGORIES:
        # the following statement only selects images that belongs only to one class and a confidence of 1.0
        cursor.execute("""INSERT INTO SelectedImages 
                      SELECT img.ImageID
                      FROM Images AS img
                      INNER JOIN Labels AS l ON l.ImageID = img.ImageID
                      WHERE img.ImageID NOT IN (SELECT ImageID FROM SelectedImages)
                        AND l.Confidence = 1.0
                        AND l.LabelName IN (SELECT LabelName FROM Dict WHERE DisplayLabelName = ?)
                        AND img.Subset = ?
                        AND img.Thumbnail300KURL IS NOT NULL
                      LIMIT ?""", (elem, mode, count))
        print("Collected all {}-set images for class ".format(mode), elem)


def clean_up_database(conn, category_list):
    print("Executing some cleanup on the database")
    cursor = conn.cursor()
    cursor.execute("""pragma journal_mode=OFF""")
    sql = "DELETE FROM Dict WHERE DisplayLabelName NOT IN ({seq})".format(
        seq=','.join(['?'] * len(category_list)))
    cursor.execute(sql, category_list)
    for i, category in enumerate(category_list):
        cursor.execute("""UPDATE Dict SET ClassNumber = ? WHERE DisplayLabelName = ?""", (i, category))

    cursor.execute("""DELETE FROM Images WHERE ImageID NOT IN (SELECT * FROM SelectedImages)""")

    cursor.execute("""DELETE FROM Labels WHERE ImageID NOT IN (SELECT * FROM SelectedImages)
                       OR LabelName NOT IN (SELECT LabelName FROM Dict) OR Confidence = 0.0""")

    cursor.execute("""DELETE FROM SelectedImages""")
    conn.commit()

    conn.cursor().execute("""VACUUM""")
    conn.cursor().execute("""pragma journal_mode=DELETE""")
    conn.commit()


def download_all():
    with sqlite3.connect(config.DATABASE['filename']) as conn:
        image_list = conn.cursor().execute('''SELECT I.ImageID, I.Thumbnail300KURL, I.OriginalURL, 
                                                    D.DisplayLabelName, I.Subset
                                              FROM Images AS I
                                              INNER JOIN Labels AS L ON I.ImageID = L.ImageID
                                              INNER JOIN Dict AS D ON L.LabelName = D.LabelName''')
        image_list = image_list.fetchall()

        create_dirs()

        pool = multiprocessing.Pool(processes=config.THREAD_COUNT)
        failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, image_list), total=len(image_list)))
        time.sleep(1)
        print("Download completed with {} failures".format(failures))


def create_dirs():
    directory = config.DATA_DIRECTORY + "Images/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    for category in config.CATEGORIES:
        directory = config.DATA_DIRECTORY + "Images/" + category
        if not os.path.exists(directory):
            os.mkdir(directory)

        for subset in ['/train', '/validation', '/test']:
            directory_new = directory + subset
            if not os.path.exists(directory_new):
                os.mkdir(directory_new)


class RedirectFilter(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            return None


def download_image(image_row):
    (image_id, url, origUrl, category, subset) = image_row
    file_extension = origUrl.split('.')[-1]
    filename = config.DATA_DIRECTORY + "Images/{}/{}/{}.{}".format(category, subset, image_id, file_extension)

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        opener = urllib.request.build_opener(RedirectFilter)
        response = opener.open(url)
        image_data = response.read()
    except:
        # print('Warning: Could not download image {} from {}'.format(image_id, url))
        return 1

    try:
        output = open(filename, "wb")
        output.write(image_data)
        output.close()
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0


def setup():
    download_files()
    build_database()
    shutil.rmtree(config.DATA_DIRECTORY + "/2017_11")
    produce_subset()
    download_all()


if __name__ == '__main__':
    setup()

