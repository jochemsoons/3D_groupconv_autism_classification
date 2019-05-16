def explore_data(file):
    print("Printing description of data contents...")
    summaries = file['summaries']
    print("Data consists of {} summaries:\n {}".format(len(summaries), list(summaries)))
    attrs = summaries.attrs
    labels = attrs['DX_GROUP']
    print("Data consists of {} samples".format(len(labels)))
    patients = 0
    controls = 0
    for label in labels:
        if label == 1:
            patients += 1
        else:
            controls += 1
    print("ASD patients: {} \nControl group: {}".format(patients, controls))



