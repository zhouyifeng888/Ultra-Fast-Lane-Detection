
import math

import cv2
import scipy
import numpy as np
from sklearn.linear_model import LinearRegression

from mindspore.common.initializer import (Normal, One, Uniform, Zero)


def initialize_weight_goog(shape=None, layer_type='conv', bias=False):
    if layer_type not in ('conv', 'bn', 'fc'):
        raise ValueError(
            'The layer type is not known, the supported are conv, bn and fc')
    if bias:
        return Zero()
    if layer_type == 'conv':
        assert isinstance(shape, (tuple, list)) and len(
            shape) == 3, 'The shape must be 3 scalars, and are in_chs, ks, out_chs respectively'
        n = shape[1] * shape[1] * shape[2]
        return Normal(math.sqrt(2.0 / n))
    if layer_type == 'bn':
        return One()
    assert isinstance(shape, (tuple, list)) and len(
        shape) == 2, 'The shape must be 2 scalars, and are in_chs, out_chs respectively'
    n = shape[1]
    init_range = 1.0 / math.sqrt(n)
    return Uniform(init_range)


def print_trainable_params_count(network):
    params = network.trainable_params()
    trainable_params_count = 0
    for i in range(len(params)):
        param = params[i]
        shape = param.data.shape
        size = np.prod(shape)
        trainable_params_count += size
    print("trainable_params_count:" + str(trainable_params_count))


class TusimpleAccEval(object):
    lr = LinearRegression()
    pixel_thresh = 20

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            TusimpleAccEval.lr.fit(ys[:, None], xs)
            k = TusimpleAccEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta
    
    @staticmethod
    def generate_tusimple_lines(out, shape, griding_num, localization_type='rel'):

        out_loc = np.argmax(out, axis=0)

        if localization_type == 'rel':
            prob = scipy.special.softmax(out[:-1, :, :], axis=0)
            idx = np.arange(griding_num)
            idx = idx.reshape(-1, 1, 1)

            loc = np.sum(prob * idx, axis=0)

            loc[out_loc == griding_num] = griding_num
            out_loc = loc
        lanes = []
        for i in range(out_loc.shape[1]):
            out_i = out_loc[:, i]
            lane = [int(round((loc + 0.5) * 1280.0 / (griding_num - 1)))
                    if loc != griding_num else -2 for loc in out_i]
            lanes.append(lane)
        return lanes
    
    @staticmethod
    def bench(pred, gt, y_samples):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        angles = [TusimpleAccEval.get_angle(
            np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [TusimpleAccEval.pixel_thresh /
                   np.cos(angle) for angle in angles]
        line_accs = []
        for x_gts, thresh in zip(gt, threshs):
            accs = [TusimpleAccEval.line_accuracy(
                np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            line_accs.append(max_acc)
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.)

def count_im_pair(anno_lanes, detect_lanes, sim_threshold=0.5):
	if len(anno_lanes):
		return (0, len(detect_lanes), 0, 0)

	if len(detect_lanes):
		return (0, 0, 0, len(anno_lanes))
    
    similarity = []
    for _ in range(len(anno_lanes)):
        similarity.append(np.zeros(len(detect_lanes)))
    similarity = np.array(similarity).astype(np.int64)
    
    
    for i in range(len(anno_lanes)):
		curr_anno_lane = anno_lanes[i]
        for j in range(len(detect_lanes)):
			curr_detect_lane = detect_lanes[j];
			similarity[i][j] = get_lane_similarity(curr_anno_lane, curr_detect_lane);
    
    m = len(similarity);
	n = len(similarity[0])
    have_exchange = false;
    if (m > n) {
        have_exchange = true;
        tmp = m
        m = n
        n = tmp
    }
    
    gra = PipartiteGraph(m , n)
    
    for i in range(gra.leftNum):
        for j in range(gra.rightNum):
			if(have_exchange)
				gra.mat[i][j] = similarity[j][i];
			else
				gra.mat[i][j] = similarity[i][j];
        
    gra.match();
    match1 = gra.leftMatch;
    match2 = gra.rightMatch;
    if have_exchange:
        tmp = match1
        match1 = match2
        match2 = tmp
    anno_match = match1

	
	curr_tp = 0;
	for i in range(anno_lanes.size()):
		if anno_match[i]>=0 and similarity[i][anno_match[i]] > sim_threshold:
			curr_tp++
		else:
			anno_match[i] = -1
	
	curr_fn = len(anno_lanes) - curr_tp
	curr_fp = len(detect_lanes) - curr_tp
	return (curr_tp, curr_fp, 0, curr_fn)



#def makeMatch(const vector<vector<double> > &similarity, vector<int> &match1, vector<int> &match2):
#	m = len(similarity);
#	n = len(similarity[0])
#    have_exchange = false;
#    if (m > n) {
#        have_exchange = true;
#        tmp = m
#        m = n
#        n = tmp
#    }
#    
#    gra = PipartiteGraph(m , n)
#    
#    for i in range(gra.leftNum):
#        for j in range(gra.rightNum):
#			if(have_exchange)
#				gra.mat[i][j] = similarity[j][i];
#			else
#				gra.mat[i][j] = similarity[i][j];
#        
#    gra.match();
#    match1 = gra.leftMatch;
#    match2 = gra.rightMatch;
#    if have_exchange:
#        tmp = match1
#        match1 = match2
#        match2 = tmp
        
def get_lane_similarity(lane1, lane2, lane_width=30):
    
	if(len(lane1)<2 || len(lane2)<2):
		return 0;
    
	im1 = np.zeros((im_height, im_width)).astype(np.uint8)
	im2 = np.zeros((im_height, im_width)).astype(np.uint8)
    
	if len(lane1)==2:
		p_interp1 = lane1;
	else:
		p_interp1 = splineInterpTimes(lane1, 50);

	if len(lane2)==2:
		p_interp2 = lane2;
	else:
		p_interp2 = splineInterpTimes(lane2, 50);
	
	for n in range(len(p_interp1)-1):
		cv2.line(im1, p_interp1[n], p_interp1[n+1], 1, lane_width)
	for n in range(len(p_interp2)-1):
		cv2.line(im2, p_interp2[n], p_interp2[n+1], 1, lane_width)

	sum_1 = im1.sum()
	sum_2 = im2.sum()
	inter_sum = (im1*im2).sum()
	union_sum = sum_1 + sum_2 - inter_sum; 
	iou = inter_sum / union_sum;
	return iou;

vector<Point2f> Spline::splineInterpTimes(tmp_line, times):
    res= [];

    if len(tmp_line) == 2:
        double x1 = tmp_line[0][0];
        double y1 = tmp_line[0][1];
        double x2 = tmp_line[1][0];
        double y2 = tmp_line[1][1];

        for k in range(times):
            double xi =  x1 + double((x2 - x1) * k) / times;
            double yi =  y1 + double((y2 - y1) * k) / times;
            res.append((xi, yi));
        
    elif len(tmp_line) > 2:
        tmp_func = cal_fun(tmp_line);
        if len(tmp_func)<=0 {
            print("in splineInterpTimes: cal_fun failed")
            return res;
        }
        for j in range(len(tmp_func)):
            delta = tmp_func[j]['h'] / times;
            for k in range(len(times)):
                double t1 = delta*k;
                double x1 = tmp_func[j]['a_x'] + tmp_func[j]['b_x']*t1 + tmp_func[j]['c_x']*math.pow(t1,2) + tmp_func[j]['d_x']*math.pow(t1,3)
                double y1 = tmp_func[j]['a_y'] + tmp_func[j]['b_y']*t1 + tmp_func[j]['c_y']*math.pow(t1,2) + tmp_func[j]['d_y']*math.pow(t1,3)
                res.append((x1, y1));
        
        res.append(tmp_line[len(tmp_line) - 1])
	else:
		print("in splineInterpTimes: not enough points")
    return res;

def cal_fun(point_v)
{   
    n = len(point_v)
    func_v = [{}]*(n-1)
    
    if(n<=2) {
        print("in cal_fun: point number less than 3")
        return func_v
    }

    Mx = [0.0]*n
    My = [0.0]*n
    A = [0.0]*(n-2)
    B = [0.0]*(n-2)
    C( = [0.0]*(n-2)
    Dx = [0.0]*(n-2)
    Dy = [0.0]*(n-2)
    h = [0.0]*(n-2)

    for i in range(n-1):
        h[i] = math.sqrt(math.pow(point_v[i+1][0] - point_v[i][0], 2) + math.pow(point_v[i+1][1] - point_v[i][1], 2));

    for i in range(n-2):
        A[i] = h[i];
        B[i] = 2*(h[i]+h[i+1]);
        C[i] = h[i+1];

        Dx[i] =  6*( (point_v[i+2][0] - point_v[i+1][0])/h[i+1] - (point_v[i+1][0] - point_v[i][0])/h[i] );
        Dy[i] =  6*( (point_v[i+2][1] - point_v[i+1][1])/h[i+1] - (point_v[i+1][1] - point_v[i][1])/h[i] );

    C[0] = C[0] / B[0];
    Dx[0] = Dx[0] / B[0];
    Dy[0] = Dy[0] / B[0];
    
    i =1 
    while i<n-2:
        double tmp = B[i] - A[i]*C[i-1];
        C[i] = C[i] / tmp;
        Dx[i] = (Dx[i] - A[i]*Dx[i-1]) / tmp;
        Dy[i] = (Dy[i] - A[i]*Dy[i-1]) / tmp;
        i+=1
        
    Mx[n-2] = Dx[n-3];
    My[n-2] = Dy[n-3];
    i = n-4
    while i>=0:
        Mx[i+1] = Dx[i] - C[i]*Mx[i+2];
        My[i+1] = Dy[i] - C[i]*My[i+2];
        i = i-1

    Mx[0] = 0;
    Mx[n-1] = 0;
    My[0] = 0;
    My[n-1] = 0;

    for i in range(n-1):
        func_v[i]['a_x'] = point_v[i][0]
        func_v[i]['b_x'] = (point_v[i+1][0] - point_v[i][0])/h[i] - (2*h[i]*Mx[i] + h[i]*Mx[i+1]) / 6
        func_v[i]['c_x'] = Mx[i]/2
        func_v[i]['d_x'] = (Mx[i+1] - Mx[i]) / (6*h[i])

        func_v[i]['a_y'] = point_v[i][1]
        func_v[i]['b_y'] = (point_v[i+1][1] - point_v[i][1])/h[i] - (2*h[i]*My[i] + h[i]*My[i+1]) / 6
        func_v[i]['c_y'] = My[i]/2
        func_v[i]['d_y'] = (My[i+1] - My[i]) / (6*h[i]);

        func_v[i]['h'] = h[i]
    
    return func_v;
}
            
            
class PipartiteGraph:
    def __init__(self,int leftNum, int rightNum):
        self.leftNum = leftNum
        self.rightNum = rightNum
        self.leftMatch = [-1]*leftNum
        self.rightMatch = [-1]*rightNum
        self.leftUsed = [False]*leftNum
        self.rightUsed = [False]*rightNum
        self.leftWeight = [0.0]*leftNum
        self.rightWeight = [0.0]*rightNum
        self.mat = []
        for i in range(leftNum):
            self.mat.append([0.0]*rightNum)
    
    def matchDfs(u):
        self.leftUsed[u] = True;
        for v in range(self.rightNum):
            if (not self.rightUsed[v]) and  (math.fabs(self.leftWeight[u] + self.rightWeight[v] - self.mat[u][v]) < 1e-2):
                self.rightUsed[v] = True;
                if self.rightMatch[v] == -1 or matchDfs(self.rightMatch[v]):
                    self.rightMatch[v] = u;
                    self.leftMatch[u] = v;
                    return True
        return False;
    
    def match():
        
        for i in range(self.leftNum):
            self.leftWeight[i] = -1e5;
            for j in range(self.rightNum):
                if self.leftWeight[i] < self.mat[i][j] 
                    self.leftWeight[i] = self.mat[i][j]

        for u in range(self.leftNum):
            while (True) 
                if matchDfs(u):
                    break;
                    
                d = 1e10;
                for i in range(self.leftNum):
                    if self.leftUsed[i]:
                        for j in range(self.rightNum):
                            if not self.rightUsed[j]:
                                d = min(d, self.leftWeight[i] + self.rightWeight[j] - self.mat[i][j]);
                        
                if d == 1e10:
                    return
                for i in range(self.leftNum):
                    if self.leftUsed[i]:
                        self.leftWeight[i] -= d;
                for i in range(self.rightNum):
                    if self.rightUsed[i]:
                        self.rightWeight[i] += d;
        
