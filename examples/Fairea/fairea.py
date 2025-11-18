import sys
sys.path.append("../../")
from collections import defaultdict
from callmefair.util.fair_util import calculate_fairness_score
import numpy as np
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Polygon, Point, LineString
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

def get_classifier(name):
    """ Creates a default classifier based on name.

    Parameters:
        name (str) -- Name of the classifier
    Returns:
        clf (classifer) -- Classifier with default configuration from scipy
    """
    if name == "lr":
        clf = LogisticRegression(max_iter=200, solver="saga")
    elif name == "dt":
        clf = tree.DecisionTreeClassifier()
    elif name == "svm":
        clf = svm.SVC(probability=True)
    elif name == "bayes":
        clf = GaussianNB()
    elif name == "tabular":
        clf = TabNetClassifier(seed=42)
    elif name == "cat":
        clf = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, thread_count=-1)
    return clf


def create_baseline(clf_name,dataset_orig, privileged_groups,unprivileged_groups,
                    data_splits=20,repetitions=20,odds={"0":[1,0],"1":[0,1]},options = [0,1],
                   degrees = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],verbose=False):
    """ Create a baseline by mutating predictions of an original classification model (clf_name).

    Parameters:
        clf_name (str)          -- Name of the original classifier to mutate
        dataset_orig (dataset)  -- Dataset used for training and testing
        privileged_groups (list) -- Attribute and label of privileged group
        unprivileged_groups(list)--Attribute and label of unprivileged group
        data_splits (int)       -- Number of different datasplits 
        repetitions (int)       -- Number of repetitions of mutation process for each datasplit
        odds (dict)             -- Odds for mutation. Keys determine the "name" of mutation strategy, values the odds for each label 
        options (list)          -- Available labels to mutate predictions
        degrees (list)          -- Mutation degrees that are used to create baselines
        verbose (bool)          -- Outputs number of current datasplit
        
    Returns:
        results (dict) -- dictionary of mutation results (one entry for each key in odds)
            dictonary values are list (mutation degree) of lists (performance for each datasplit X repetitions)
    """
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    ids = [x for x in range(len(dataset_orig_test.labels))]
    l = len(dataset_orig_test.labels)
    
    results = defaultdict(lambda: defaultdict(list))
    
    
    # Iterate over different datasplits
    for s in range(data_splits):
        if verbose:
            print ("Current datasplit:",s)
        np.random.seed(s)
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        scaler = StandardScaler()
        dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
        dataset_orig_test.features = scaler.transform(dataset_orig_test.features)
        
        # Make initial predictions
        clf = get_classifier(clf_name)
        clf = clf.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())
        pred = clf.predict(dataset_orig_test.features).reshape(-1,1)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.labels = pred
        
        # Mutate labels for each degree
        for degree in degrees:
            # total number of labels to mutate
            to_mutate = int(l*degree)

            for name,o in odds.items():
                # Store each mutation attempt
                hist = []
                for _ in range(repetitions):
                    # Generate new random labels
                    rand = np.random.choice(options, to_mutate, p=o)
                    # Select prediction ids that are being mutated
                    to_change = np.random.choice(ids, size=to_mutate, replace=False)
                    changed = np.copy(pred)
                    for t,r in zip(to_change, rand):
                        changed[t] = r
                    
                    # Determine accuray and fairness of mutated model 
                    dataset_orig_test_pred.labels = changed
                    class_metric = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                     unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                    stat = class_metric.statistical_parity_difference()
                    aod = class_metric.average_abs_odds_difference()
                    eod = class_metric.equal_opportunity_difference()
                    di = class_metric.disparate_impact()
                    ti = class_metric.theil_index()
                    f_score = calculate_fairness_score(eod, aod, stat, di, ti)['overall_score']
                    balanced_acc = 0.5 * (class_metric.true_positive_rate() + class_metric.true_negative_rate())
                    hist.append([balanced_acc,stat,aod,f_score])
                results[name][degree] += hist
    return results



def normalize(base_accuray, base_fairness, method_dict=dict()):
    """ Normalize baseline and bias mitigation methods within the range of the baseline.
    
    Fairness normalization: keeps the same behavior (normalized to baseline range)
    Accuracy normalization: inverted behavior but ensures 0-1 scale proportional to baseline

    Parameters:
        base_accuray (list)  -- Accuracy at each mutation degree
        base_fairness (list) -- Fairness at each mutation degree
        method_dict (dict)   -- Accuracy and fairness of bias mitigation methods
        
    Returns:
        normalized_accuracy (list) -- Normalized accuracy at each mutation degree (inverted)
        normalized_accuracy (list) -- Normalized fairness at each mutation degree
        normalized_methods (list) -- Normalized accuracy and fairness of bias mitigation methods (0-1 scale)
    """

    # Determine range of values for baseline
    range_accuracy = np.max(base_accuray)-np.min(base_accuray)
    range_fairness = np.max(base_fairness)-np.min(base_fairness)
    min_accuracy = np.min(base_accuray)
    max_accuracy = np.max(base_accuray)
    min_fairness = np.min(base_fairness)
    
    # Normalize fairness values (keep same behavior)
    normalized_fairness = (base_fairness-min_fairness)/range_fairness
    
    # Normalize accuracy values (inverted behavior)
    # Higher accuracy becomes lower normalized value
    normalized_accuracy = (max_accuracy - base_accuray)/range_accuracy
    
    # For methods, we need to determine the overall range including both baseline and methods
    all_accuracies = list(base_accuray) + [acc for acc, fair in method_dict.values()]
    all_fairness = list(base_fairness) + [fair for acc, fair in method_dict.values()]
    
    # Determine global ranges for proper 0-1 scaling
    global_min_accuracy = np.min(all_accuracies)
    global_max_accuracy = np.max(all_accuracies)
    global_range_accuracy = global_max_accuracy - global_min_accuracy
    
    global_min_fairness = np.min(all_fairness)
    global_max_fairness = np.max(all_fairness)
    global_range_fairness = global_max_fairness - global_min_fairness
    
    # Normalize values of bias mitigation methods with proper 0-1 scaling
    normalized_methods = dict()
    for k, (acc, fair) in method_dict.items():
        # For accuracy: invert the relationship but ensure 0-1 scale
        # Higher accuracy relative to global max becomes lower normalized value
        if global_range_accuracy > 0:
            norm_acc = (global_max_accuracy - acc) / global_range_accuracy
        else:
            norm_acc = 0.0  # Handle case where all accuracies are the same
            
        # For fairness: keep proportional to global range
        if global_range_fairness > 0:
            norm_fair = (fair - global_min_fairness) / global_range_fairness
        else:
            norm_fair = 0.0  # Handle case where all fairness values are the same
            
        normalized_methods[k] = (norm_acc, norm_fair)
    
    return normalized_accuracy, normalized_fairness, normalized_methods



def classify_region(base, normalized_methods):
    """ Determine bias mitigation region of normalized bias mitigation methods.

    Parameters:
        base (LineString)  -- Geometrical line (LineString) of normalized baseline created with shapely
        normalized_methods (dict) -- Normalized accuracy and fairness of bias mitigation methods
        
    Returns:
        mitigation_regions (dict) -- Bias mitigation region for each normalized bias mitigation method
    """
    mitigation_regions = dict()
    for k,(acc,fair) in normalized_methods.items():
        # define a point for each bias mitigation method
        p = Point(fair,acc)
        # Extend bias mitigation point towards four directions (left,right,up,down)
        line_down = LineString([(p.x, p.y),(p.x, 0)])
        line_right = LineString([(p.x, p.y),(2, p.y)])
        line_up = LineString([(p.x, p.y),(p.x, 2)])
        line_left = LineString([(p.x, p.y),(0, p.y)])
        # Determine bias mitigation region based on intersection with baseline
        if base.intersects(line_down) and base.intersects(line_right):
            mitigation_regions[k] = "good trade-off"
        elif base.intersects(line_down):
            mitigation_regions[k] = "win-win"
        elif base.intersects(line_up):
            mitigation_regions[k] = "bad trade-off"
        elif base.intersects(line_left):
            mitigation_regions[k] = "lose-lose"
        elif fair < 0:
            mitigation_regions[k] = "lose-lose"
        else:
            mitigation_regions[k] = "inverted"
    return mitigation_regions

def cut(line, distance):
    """ Cuts a line in two parts, at a distance from its starting point

    Parameters:
        line (LineString)  -- Geometrical line (LineString) of to be cut, created with shapely
        distance (float) -- Distance from origin (first point) of line where the cut should be place
        
    Returns:
        LineString,LineString -- Left and right part of original line, cut at the specified distance
    """
    # Check whether line cut is possible
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)

    # Iterate each point of line (Line = (point1, point2 ...)) to find position of cut
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

def compute_area(base,method):
    """ Compute area a bias mitigation method forms with the baseline, 
        by connection them with a horizontal and vertical line.

    Parameters:
        base (LineString)  -- Geometrical line (LineString) of normalized baseline created with shapely
        method (tuple)     -- Normalized accuracy and fairness of a bias mitigation method
        
    Returns:
        area (float) -- Bias mitigation region for each normalized bias mitigation method
    """

    # Create Point for bias mitigation method performance
    acc,fair = method
    p = Point(fair,acc)

    # Create horizontal and vertical line to connect point with baseline
    line_down = LineString([(p.x, p.y),(p.x, 0)])
    line_right = LineString([(p.x, p.y),(1, p.y)])

    # find intersection
    down_inter = base.intersection(line_down)
    right_inter = base.intersection(line_right)
    
    # Create a Polygon with the bias mitigation method point, and the intersections with the baseline
    cut_right,cut_left=cut(base,base.project(down_inter))
    cut_right,cut_left=cut(cut_right,base.project(right_inter))
    area = [(p.x,p.y)] + list(cut_left.coords) + [(p.x,p.y)]
    poly = Polygon(area)
    return poly.area