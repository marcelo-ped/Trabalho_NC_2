import os
from nn_pso import execute_pso
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nn_ga import execute_ga


parser = argparse.ArgumentParser()
parser.add_argument('-algorithm', dest = 'algorithm', help = 'Type 0 for execute PSO algorithm or Type 1 for execute GA algorithm', required = True, type = str)
parser.add_argument('-dataset', dest = 'dataset', help = 'Type 0 for use Iris dataset or type 1 for use Wine dataset or type 2 to use Breast Cancer dataset', required = True, type=str)
args = parser.parse_args()
algorithm = args.algorithm
dataset = args.dataset
if algorithm == "0":
    time_seconds , cost, acc_test, f1_test, prec_test, rec_test, acc_train, f1_train, prec_train, rec_train = execute_pso(int(dataset))
else:
    time_seconds , cost, acc_test, f1_test, prec_test, rec_test, acc_train, f1_train, prec_train, rec_train = execute_ga(int(dataset))
print("TREINO MÉTRICAS")
print("Acurácia "+ str(acc_train))
print("F1-score "+str(f1_train))
print("Precisão "+ str(prec_train))
print("Recall " + str(rec_train))
print("TESTE MÉTRICAS")
print("Acurácia "+ str(acc_test))
print("F1-score "+str(f1_test))
print("Precisão "+ str(prec_test))
print("Recall " + str(rec_test))
print("TEMPO DE EXECUÇÃO")
print(time_seconds, " segundos")
