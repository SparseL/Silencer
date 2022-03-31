from sp_danmf import spDANMF
from parameter_parser import parameter_parser
from sklearn import metrics
import pickle
''''''

def main(args,graph,label):

    model = spDANMF(graph, args, label)
    model.pre_training(8)
    res = model.training()
    NMI = metrics.normalized_mutual_info_score(res,label)
    print('dataSet:{}; noise intensities p={}; NMI:{}'.format(args.dataSetName,args.ratio,NMI))

if __name__ == "__main__":
    args = parameter_parser()
    args.ratio = 0.01 # noise intensities p=0.01
    args.graphPath='..//input//email-Eu-core//email-Eu-core-ratio_{}.graph'.format(args.ratio)
    args.labelPath='..//data//email-Eu-core.label'
    with open(args.labelPath,'rb')as f:
        label = pickle.load(f)
    with open(args.graphPath, 'rb')as f:
        graph = pickle.load(f)

    main(args,graph,label)