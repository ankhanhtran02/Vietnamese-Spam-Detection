import pandas as pd
import numpy as np
import re

def contains_weird_char(text:str):
    vietnamese_pattern = r'[0-9a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý\s|_]' # If we don't need special ascii characters, we can add \W
    vietnamese_chars = re.compile(vietnamese_pattern)
    # Count the non-Vietnamese characters using regex
    weird_chars = vietnamese_chars.sub('', text)
    # print(weird_chars)
    return len(weird_chars) > 4

def contains_links(text:str):
    link_pattern = r'[a-zA-Z0-9]+\.[a-z]+\/([a-zA-Z0-9]+)?|[a-zA-Z0-9]+\.com|https?:\/\/[a-zA-Z0-9\.\/]+|www.[a-zA-Z0-9\.\/]+'
    pattern = re.compile(link_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count > 0

def contains_phone_numbers(text:str):
    phone_pattern = r'\+?[0-9]{11,12}|0[0-9]{3}[\. -]?[0-9]{3}[\. -]?[0-9]{3}|0[0-9]{3}[\. -]?[0-9]{2}[\. -]?[0-9]{2}[\. -]?[0-9]{2}|0[0-9]{2}[\. -]?[0-9]{3}[\. -]?[0-9]{4}|(18|19)00[0-9]{4}'
    pattern = re.compile(phone_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count > 0

def contains_weird_capitalization(text:str):
    cap_pattern = r'[a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽềềểếễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹý]*[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý]*'
    pattern = re.compile(cap_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count > 0

def contains_money(text:str):
    money_pattern = r'([0-9]+[\.,])*[0-9]+([Kkdđ]| ngàn| nghìn|.ooo|tr| triệu| vnd| ty| tỷ| đồng| tỉ| ti)'
    pattern = re.compile(money_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count > 0

def BaselineSystem(filename='datasets\\test_set.csv'):
    df = pd.read_csv(filename)
    df = df[df.columns[:2]]
    x_test = df['text'].astype(str)
    y_test = df['spam']
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y_predict = []
    for i in range(len(x_test)):
        x = x_test[i]
        if contains_weird_char(x) + contains_links(x) + contains_money(x) + contains_phone_numbers(x) + contains_weird_capitalization(x) >= 1:
            pred = 1
        else:
            pred = 0
        y_predict.append(pred)
        if pred == 1 and y_test[i] == 1:
            TP += 1
        elif pred == 1 and y_test[i] == 0:
            FP += 1
        elif pred == 0 and y_test[i] == 0:
            TN += 1
        elif pred == 0 and y_test[i] == 1:
            FN += 1
    predict = np.array(y_predict)
    '''Accuracy = (TP+TN)/(TP+TN+FP+FN)'''
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    '''Precision = TP/(TP+FP)'''
    precision = TP/(TP+FP)
    '''Recall = TP/(TP+FN)'''
    recall = TP/(TP+FN)
    '''F1 = 2*precision*recall/(precision + recall)'''
    f1 = 2*precision*recall/(precision + recall)
    return predict, accuracy, precision, recall, f1, TP, TN, FP, FN

if __name__ == '__main__':
    predict, accuracy, precision, recall, f1, TP, TN, FP, FN = BaselineSystem()
    print('Testing Set Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
