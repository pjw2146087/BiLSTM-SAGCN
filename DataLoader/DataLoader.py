
from .Dataset import *
from .utils import crop_trace , crop_adj

def tracedata_reader(dataset):
    def reader():
        if dataset.mode in ['train', 'eval']:
            for i in range(dataset.__len__()):
                points, labels,trace_id = dataset.__getitem__(i)
                yield (points, labels),trace_id
        else:
            for i in range(dataset.__len__()):
                points,trace_id = dataset.__getitem__(i)
                yield points ,trace_id
    return reader

def TraceLoader(dataset,max_len=5000,drop_rate=0):
    def batch_reader():
        r = tracedata_reader(dataset)()
        if dataset.mode in ['train', 'eval']:
            for item ,trace_id in r:
                cropped_Points=crop_trace(item[0],max_len,drop_rate)
                cropped_labels=crop_trace(item[1],max_len,drop_rate)
                trace_id=trace_id[0]
                cropped_adjs=crop_adj(dataset.__getadj__(trace_id),max_len,drop_rate)
                for Points,labels,adj in zip(cropped_Points,cropped_labels,cropped_adjs):
                    yield ((Points,labels),adj)
        else:
            for item ,trace_id in r:
                cropped_Points=crop_trace(item ,max_len,drop_rate)
                trace_id=trace_id[0]
                cropped_adjs=crop_adj(dataset.__getadj__(trace_id),max_len,drop_rate)
                for Points,adj in zip(cropped_Points,cropped_adjs):
                    yield (Points,adj)
    return batch_reader