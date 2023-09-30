"""This approach uses a deterministic text template to convert each row
from the restaurant dataset into a text which will be used as an input
into the BERT model for fine-tuning for the classification task.
"""
import argparse
import pandas as pd

FEATURES = ['Id', 'Open Date', 'City', 'City Group', 'Type', 'P1', 'P2', 'P3', 'P4',
            'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
            'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25',
            'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',
            'P36', 'P37', 'revenue']

TEMPLATE = """Restaurant {id} is located in {city} and is of type {type}.
The city is a {city_group} type of a city. The restaurant opened on {open_date}.
The values for the features P1 to P37 are as follows:
{p_properties}
"""
def construct_description_from_row(row: pd.Series) -> str:
    return TEMPLATE.format(id=row["Id"],
                           city=row["City"],
                           type=row["Type"],
                           city_group=row["City Group"],
                           open_date=row["Open Date"],
                           p_properties="\n".join([f"P{i}: {row[f'P{i}']}" for i in range(1, 38)]))


def main():
    parser = argparse.ArgumentParser("csv-to-text")
    parser.add_argument("-d", "--data-file",
                        type=str,
                        help="Path to the CSV with the data",
                        default="train_clf.csv")
    parser.add_argument("-o", "--output-file",
                        type=str,
                        help="Path to the output file",
                        default="train_clf.txt")

    args = parser.parse_args()
    df = pd.read_csv(args.data_file, header=0)

    text_descriptions = [construct_description_from_row(row) for _, row in df.iterrows()]

    with open(args.output_file, "w") as f:
        f.write("\n\n".join(text_descriptions))


if __name__ == "__main__":
    main()
