import logging
import numpy as np
import csv
logging.basicConfig(level=logging.INFO)

file_input='./dataset.csv'
file_output='./submission_own.csv'


class Classifier(object):
    
    def __init__(self,path):
        self.label=[0,-1,-2,-3]
        self.data_dict = {}
        self.load(path)
        self.d_num = len(self.data_dict)
        self.data_res=[]


    def load(self, path):
        a=1
        with open(path, 'r') as f:
            f.readline()
            for line in f:
                #print(line.strip().split(','))
                termNum,distNum,blockNum,Time,powerNum,Power=line.strip().split(',')
                self.data_dict[a] = [termNum,distNum,blockNum,Time,powerNum,Power]
                a=a+1
    
    def output(self,path):
        header = ['termNum','distNum','blockNum','Time','powerNum','Label']
        with open(path,'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.data_res)

    def method_test(self):
        self.data_res=[]
        for i in range(1, self.d_num + 1):
            row=self.data_dict[i]
            row[5]=np.random.choice(self.label)
            self.data_res.append(row)
    
    def method_1(self):
        logging.info('run method_1...')






def main():
    logging.info('run_classifier is running...')
    cf=Classifier(path=file_input)
    cf.method_test()
    cf.output(file_output)

    #print(len(cf.data_res))

    logging.info('run_classifier finished!')


if __name__=='__main__':
    main()