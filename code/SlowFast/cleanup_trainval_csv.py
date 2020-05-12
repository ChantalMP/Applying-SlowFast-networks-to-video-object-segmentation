from pathlib import Path


def clean_csv(csv_path, valid_filenames):
    final_csv = []

    with csv_path.open() as f:
        for idx, row in enumerate(f.readlines()):
            if idx == 0:
                final_csv.append(row)
            else:
                video_id = row.split(' ')[0]

                if video_id in valid_filenames:
                    final_csv.append(row)

    with csv_path.open('w') as f:
        f.writelines(final_csv)


if __name__ == '__main__':
    frames_path = Path('data/ava/frames')
    valid_filenames = {elem.name for elem in frames_path.glob('*')}

    paths = Path('data/ava/frame_lists').glob('*.csv')

    for path in paths:
        clean_csv(path, valid_filenames)
