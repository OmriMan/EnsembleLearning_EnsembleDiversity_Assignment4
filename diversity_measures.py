# -*- coding: utf-8 -*-
import numpy as np  
import pandas as pd  

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt  
from sklearn.tree import plot_tree 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

import random
from tabulate import tabulate
"""
Kohavi Wolpert
"""
def kohavi_wolpert(s,m):
  '''
  s - the original training set tuple of (X_train,y_train) --> ((dataframe),((dataframe))
  s[0] = x train
  s[1] = y train
  m - list of classifiers examined
  '''
  x = s[0]
  y=s[1]
  predictions =[]
  sum =0
  # For each classifier, insert all its predictions (depending on the index of x) into the list .
  for classifier in m:
    preds = classifier.predict(x.values)
    predictions.append(preds)
  # Run through all the records in the training set and for each record : count the number of classifiers that made an incorrect classification(misclassified)
  l_i=0
  for (index_x, x_i),(index_y, y_i),pred_index in zip(x.iterrows(),y.iterrows(),range(len(predictions[0]))):
    l_i=0
    for model_i_prediction in predictions:
      # l_i = number of classifiers misclassified x_i,y_i
      if model_i_prediction[pred_index] != y_i.values:
        l_i+=1
      #end sum all misclassification for x_i,y_i
    #sum = sum + l_i * (len(m) - l_i)
    sum = sum + ( l_i * ( len(m)-l_i ) ) 
  # print(f"sum/(len(x)*(len(m)**2)) : {sum/(len(x)*(len(m)**2))}")
  return (sum) / ( len(y) * (len(m)**2) )

"""
Inter Rater Measure
"""

def inter_rater_measure(s,m):
  '''
  s - the original training set tuple of (X_train,y_train) --> ((dataframe),((dataframe))
  s[0] = x train
  s[1] = y train
  m - list of classifiers examined
  '''
  x = s[0]
  y=s[1]
  predictions =[]
  sum =0
  # For each classifier, insert all its predictions (depending on the index of x) into the list .
  for classifier in m:
    preds = classifier.predict(x.values)
    predictions.append(preds)
  # Run through all the records in the training set and for each record : count the number of classifiers that made an incorrect classification(misclassified)
  l=[]
  for (index_x, x_i),(index_y, y_i),pred_index in zip(x.iterrows(),y.iterrows(),range(len(predictions[0]))):
    l_i=0
    for model_i_prediction in predictions:
      # l_i = number of classifiers misclassified x_i,y_i
      if model_i_prediction[pred_index] != y_i.values:
        l_i+=1
      #end sum all misclassification for x_i,y_i
    l.append(l_i)
    #sum = sum + l_i * (len(m) - l_i)
    sum = sum + ( l_i * ( len(m) - l_i) )
  # p = 1 - ( sum(l) / (|s| *|m|) )
  sum_l=0
  for val in l:
    sum_l+=val
  p = 1 - ( sum_l / (len(x)*len(m)) )
  result = 1 - (sum / ( len(x)*len(m)*(len(m)-1)*p*(1-p)  ) )
  return result

"""
General Diversity
"""

import statistics
def general_diversity(s,m):
  '''
  s - the original training set tuple of (X_train,y_train) --> ((dataframe),((dataframe))
  s[0] = x train
  s[1] = y train
  m - list of classifiers examined
  '''
  x = s[0]
  y=s[1]
  predictions =[]
  # For each classifier, insert all its predictions (depending on the index of x) into the list .
  for classifier in m:
    preds = classifier.predict(x.values)
    predictions.append(preds)
  # Run through all the records in the training set and for each record : count the number of classifiers that made an incorrect classification(misclassified)
  v=[]
  for (index_x, x_i),(index_y, y_i),pred_index in zip(x.iterrows(),y.iterrows(),range(len(predictions[0]))):
    l_i=0
    for model_i_prediction in predictions:
      # l_i = number of classifiers misclassified x_i,y_i
      if model_i_prediction[pred_index] != y_i.values:
        l_i+=1
      #end sum all misclassification for x_i,y_i
    v_i= (len(m)-l_i) / len(m)
    v.append(v_i)
  # end of run through all the records in the training set and for each record : count the number of classifiers that made an incorrect classification(misclassified)
  # compute variance of v using statistics.variance
  variance_v = statistics.variance(v)
  return variance_v

"""
Double Fault Measure
"""

def DoubleFaultMeasure(s, set_of_classifiers):
  train_set=s[0]
  sum = 0
  i_index = 0
  predictions =[]
  # For each classifier, insert all its predictions (depending on the index of x) into the list .
  for classifier in set_of_classifiers:
    preds = classifier.predict(train_set.values)
    predictions.append((classifier,preds))
  for i,model_i_predictions in predictions:
    for j,model_j_predictions in predictions[i_index:]:
      num_of_instances = 0
      if i != j: # if not the same tree
        for k in range(len(np.ravel(y_train))):
          if (model_i_predictions[k] != np.ravel(y_train)[k] and model_j_predictions[k] != np.ravel(y_train)[k]): # if both trees misclasified then add 1 to num of instances...
            num_of_instances += 1

        sum += num_of_instances
    i_index += 1
  res = (2*sum)/(len(train_set)*len(set_of_classifiers)*(len(set_of_classifiers)-1))
  return res

"""
Bagging using Diversity
"""

from sklearn.metrics import classification_report, confusion_matrix
from functools import reduce

def bagging_using_Diversity(i,t,s,d,features,target_name):
  '''
  i - a base inducer
  t - number of iterations
  s - the original training set
  d- diversity measure - { ""  :kohavi_wolpert , "" :inter_rater_measure ,"general diversity": general_diversity }
  features - list of feature names in s(the data set) that are not target features
  target_name - Name of the target attributes - 
    if target is a single feature - a string(target label)
    else - can be a list of strings
  '''
  diversity_measure = {"Double Fault Measure": DoubleFaultMeasure, "kohavi wolpert"  :kohavi_wolpert , "inter rater measure" :inter_rater_measure ,"general diversity": general_diversity }
  m_ts =[]
  #m_ts - list of tuples - each tuple in the following format : (model,number_of_misclassifieds)
  # m_ts --> [(model1,number_of_misclassifieds1),(model2,number_of_misclassifieds2),...,(model_t,number_of_misclassifieds_t)]
  for iter in range(t):
    s_tag = s.sample(frac = 1)
    m_t = i.estimators_[iter]
    x = s_tag.loc[:, features]
    y = s_tag.loc[:, [target_name]]
    preds = m_t.predict(x.values)
    # confusion_matrix(y,preds) - [[ture_positive, false_positive][false_neg,true_neg]]
    number_of_misclassifieds = confusion_matrix(y,preds)[0][1] + confusion_matrix(y,preds)[1][0]
    m_ts.append((m_t,number_of_misclassifieds))
  # end of rows 1-3
  x = s.loc[:, features]
  y = s.loc[:, [target_name]]
  # 5
  m_tag=[]
  # use reduce() and min() to find the minimum value in the second element of each tuple
  min_misclassifields = reduce(lambda x, y: min(x, y[1]), m_ts, float("inf"))
  # Find an element (model,number_of_misclassifieds) in m_ts(list of tuples).
  m_tags_optinals = [item for item in m_ts if item[1] == min_misclassifields]
  # m_tags_optinals - list of tuples [(model1,min_misclassifields),...]
  m_tag.append(m_tags_optinals[0][0])
  m_ts.remove(m_tags_optinals[0])
  # 6
  for i in range(1,int((t/2)-1)):
    results_of_d =[]
    # results_of_d - list of tuples - each tuple in the following format : (model,diversity measure value)
    if (d=="Double Fault Measure" or d=="kohavi wolpert" or d=="general diversity"):
      # need to take the model that return maximum diversity measure
      print(i)
      for temp_model,number_of_misclassifieds in m_ts:
        # m_tag_only_models = [tup[0] for tup in m_tag]
        results_of_d.append( (temp_model, diversity_measure[d]((x,y),m_tag+[temp_model])) )
      # need to find maximum value of diversity measure
      # use reduce() and min() to find the minimum value in the second element of each tuple
      # results_of_d[0] = (model,diversity measure value)
      max_d = reduce(lambda x, y: max(x, y[1]), results_of_d, float("-inf"))
      # Find an element (model,diversity measure value) in results_of_d(list of tuples).
      m_tags_optinals = [item for item in  results_of_d if item[1] == max_d]
      # m_tags_optinals[0] = (model,diversity measure value), e.g: (DecisionTreeClassifier(max_features='auto', random_state=1913246406), 0.014287578288100209)
      # m_tags_optinals[0][0] - temp_model
      # m_tags_optinals[0][1] - maximum diversity measure value , and its for m_tag with temp_model
      # we want to add only the model to m_tag so we add m_tags_optinals[0][0]
      m_tag.append(m_tags_optinals[0][0])
      # need to find m_tags_optinals[0][0] (the model) in m_ts and remove it from m_ts
      model_num_of_mis_tuple_to_remove_from_m_ts = [item for item in m_ts if item[0] == m_tags_optinals[0][0]]
      m_ts.remove(model_num_of_mis_tuple_to_remove_from_m_ts[0])
    else:
      # need to take the model that return minimum diversity measure
      print(i)
      for temp_model,number_of_misclassifieds in m_ts:
        # m_tag_only_models = [tup[0] for tup in m_tag]
        results_of_d.append( (temp_model, diversity_measure[d]((x,y),m_tag+[temp_model])) )
      # need to find maximum value of diversity measure
      # use reduce() and min() to find the minimum value in the second element of each tuple
      # results_of_d[0] = (model,diversity measure value)
      min_d = reduce(lambda x, y: min(x, y[1]), results_of_d, float("inf"))
      # Find an element (model,diversity measure value) in results_of_d(list of tuples).
      m_tags_optinals = [item for item in  results_of_d if item[1] == min_d]
      # m_tags_optinals[0] = (model,diversity measure value), e.g: (DecisionTreeClassifier(max_features='auto', random_state=1913246406), 0.014287578288100209)
      # m_tags_optinals[0][0] - temp_model
      # m_tags_optinals[0][1] - minimum diversity measure value , and its for m_tag with temp_model
      # we want to add only the model to m_tag so we add m_tags_optinals[0][0]
      m_tag.append(m_tags_optinals[0][0])
      # need to find m_tags_optinals[0][0] (the model) in m_ts and remove it from m_ts
      model_num_of_mis_tuple_to_remove_from_m_ts = [item for item in m_ts if item[0] == m_tags_optinals[0][0]]
      m_ts.remove(model_num_of_mis_tuple_to_remove_from_m_ts[0])

  return m_tag
