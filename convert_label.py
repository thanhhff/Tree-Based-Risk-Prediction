import csv

rows = []
with open('data/NHANESI_subset_y.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        rows.append(row)
for i, row in enumerate(rows):
    if i > 0:
        if float(row[1]) <= 0:
            rows[i][1] = '1'
        else:
            rows[i][1] = '0'

with open('label.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

#https://github.com/amanchadha/coursera-ai-for-medicine-specialization/tree/master/AI%20for%20Medical%20Prognosis/Week%204
