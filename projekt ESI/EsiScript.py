import time
import pandas as pd
from math import log2
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
from graphviz import Digraph
#load data into a DataFrame object:
data = pd.read_csv("C:\Programowanie\python\projekt ESI\EsiProjekt_kopia_bezpieczenstwa.csv",sep=";")
df = pd.DataFrame(data)

start_time = time.time()


def BinaryTreePrinter(tabelka) :  #calosc funkcji na tabie dodac
        j = True
        h = []
        tablica = []
        liczby = []
        secondlist = []
        for i in tabelka: 
                if j:
                        tabelowedrzewo = Node(i)
                        j=False
                        h.append(tabelowedrzewo)
                        liczby.append(0)
                        secondlist.append("X")
                else:
                        if i[-1] == "?":
                                while liczby[-1] == 2:
                                        liczby.pop()
                                        h.pop()
                                if liczby[-1] == 0:
                                        tablica.append( Node(i, parent=h[-1], type="tak"))
                                else :
                                        tablica.append(Node(i, parent=h[-1], type="nie"))
                                liczby.append(liczby.pop() + 1)
                                liczby.append(0)
                                h.append(tablica[-1])
                                secondlist.append("X")
                        else:
                                while liczby[-1] == 2:
                                        liczby.pop()
                                        h.pop()
                                if liczby[-1] == 0:
                                        tablica.append( Node(i, parent=h[-1], type = "tak"))
                                else :
                                        tablica.append( Node(i, parent=h[-1], type = "nie"))
                                liczby.append(liczby.pop() + 1)
                                secondlist.append("odp")
                                for ik in range(len(secondlist),0,-1):
                                     if(secondlist[ik-1] == "X"):
                                          secondlist[ik-1] = len(secondlist)
                                          break
        for pre, fill, node in RenderTree(tabelowedrzewo):
                print("%s%s" % (pre, node.name))
        edgeattrfunc=lambda parent, child:'style=bold, label="{}"'.format(child.type)
        UniqueDotExporter(tabelowedrzewo, edgeattrfunc=edgeattrfunc).to_picture("C:\Programowanie\python\graftest\graf.png")
        return secondlist

def DecisionPredictor(tablica,odpowiedzi,questions, secondlist):
    x=0
    while (tablica[x])[-1]=="?":
        index_of_question=questions.index(tablica[x])
        if odpowiedzi[index_of_question]==1:
            x=x+1
        else:
            x = secondlist[x]
    return tablica[x]
         

#do liczenia entropii warunkowej pytanie + decyzja (E)
#question_idx -> indeks pytania
#start_decision_idx -> startowy indeks decyzji (indeks wiersza Horror)
#end_decision_idx -> koncowy indeks decyzji (indeks wiersz komedio-dramat)
#chcek_true -> do potwierdzania zaprzeczania warunku
def GetConditionEntropy(data_frame,question_idx,start_decision_idx,end_decision_idx,check_true):
    length = len(data_frame.iloc[question_idx])
    set1 = list(data_frame.iloc[question_idx])
    list_of_occur=[]
    for i in range(start_decision_idx,end_decision_idx):
        decisionset = list(data_frame.iloc[i])
        list_of_occur.append(CheckHowManyRep(set1,decisionset,check_true))
    if(sum(list_of_occur)!=0):
        entropies=list(map(SimpleEntropy,map(lambda x: x/sum(list_of_occur),list_of_occur)))
        sum_of_entropies=sum(entropies)
    else:
        sum_of_entropies=0
    return [sum(list_of_occur),sum_of_entropies]

#proste liczenie entropii 
def SimpleEntropy(value):
    if value<=0:
        return value*0
    return -value*log2(value)

#funkcja pomocnicza patrzy ile jest cześci wspólnych przy potwierdzonym warunku pytania i zaprzeczenie od tego jest zmienna confirmcondition
def CheckHowManyRep(table1,table2,confirmcondition):
    mysum = 0
    if confirmcondition==True:
        for i in range(0,len(table1)):
            if table1[i]==table2[i] and table1[i]==1:
                mysum=mysum+1
    else:
        for i in range(0,len(table1)):
            if table1[i]!=table2[i] and table1[i]==0:
                mysum=mysum+1
    return mysum

#Do liczenia I entropii 
def GetEntropy(data_frame,start_idx,end_idx):
    list_of_occur=[]
    lenght = len(data_frame.iloc[start_idx])
    for i in range(start_idx,end_idx):
        decisionset = list(data_frame.iloc[i])
        list_of_occur.append(sum(decisionset))
    entropies=list(map(SimpleEntropy,map(lambda x: x/sum(list_of_occur),list_of_occur)))
    return sum(entropies)

def GetTheHighestEntropy(table,end_of_questions,start_idx,end_idx):
    I=GetEntropy(table,start_idx,end_idx)
    list_of_entropies = []
    for i in range(0,end_of_questions):
        rowsum=sum(table.iloc[i])
        if(rowsum==len(table.iloc[i])or rowsum==0):
           list_of_entropies.append(-1)
           continue
        entropy_true_condition = GetConditionEntropy(table,i,start_idx,end_idx,True)
        entropy_false_condition = GetConditionEntropy(table,i,start_idx,end_idx,False)
        sum_of_entropies=entropy_true_condition[0]+entropy_false_condition[0]
        E=(entropy_true_condition[0]/sum_of_entropies)*entropy_true_condition[1]+(entropy_false_condition[0]/sum_of_entropies)*entropy_false_condition[1]
        list_of_entropies.append(I-E) 
    return list_of_entropies.index(max(list_of_entropies))

def DivideTable(table,index_of_row):
    list_of_table = []
    condition=table.columns[table.iloc[index_of_row]==1]
    true_table=table.iloc[:, table.columns.isin(condition)]
    list_of_table.append(true_table)
    condition=table.columns[table.iloc[index_of_row]<1]
    false_table=table.iloc[:,table.columns.isin(condition)]
    list_of_table.append(false_table)
    return list_of_table


def CheckIfAnyDecisionIsMade(data_frame,start_idx,end_idx):
    for i in range(start_idx,end_idx):
        if(sum(data_frame.iloc[i])==len(data_frame.iloc[i])):
            return i
    return False

class Branch:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.is_left_branch_appear = False
        self.is_right_branch_appear = False

        
    def set_branch_appear(self, is_left_branch_appear, is_right_branch_appear):
        self.is_left_branch_appear = is_left_branch_appear
        self.is_right_branch_appear = is_right_branch_appear


def DecisionTree(data_frame,end_of_questions,start_idx,end_idx,questions):
    list_to_create_decision_tree = []
    list_of_nodes = []
    first_node = Branch(data_frame)
    list_of_nodes.append(first_node)
    while(len(list_of_nodes)>0):
        current_node = list_of_nodes[-1]
        if(current_node.is_right_branch_appear==True and current_node.is_left_branch_appear==True):
            list_of_nodes.pop()
            continue
        if(CheckIfAnyDecisionIsMade(current_node.data_frame,start_idx,end_idx)!=False):
            decision_idx = CheckIfAnyDecisionIsMade(current_node.data_frame,start_idx,end_idx)
            list_to_create_decision_tree.append(questions[decision_idx])
            list_of_nodes.pop()
        elif(current_node.is_left_branch_appear==False):
            current_node.is_left_branch_appear=True
            index_of_question=GetTheHighestEntropy(current_node.data_frame,end_of_questions,start_idx,end_idx)
            list_to_create_decision_tree.append(questions[index_of_question])
            new_node=Branch(DivideTable(current_node.data_frame,index_of_question)[0])
            list_of_nodes.append(new_node)
        elif(current_node.is_right_branch_appear==False):
            current_node.is_right_branch_appear=True
            index_of_question=GetTheHighestEntropy(current_node.data_frame,end_of_questions,start_idx,end_idx)
            new_node=Branch(DivideTable(current_node.data_frame,index_of_question)[1])
            list_of_nodes.append(new_node)
        else:
            return list_to_create_decision_tree
    return list_to_create_decision_tree

new_df=df.iloc[:,2:]
myquestions=list(df.iloc[:,1])

def GetProperties(vector):
    endOfQuestions=0
    for i in vector:
        if(i[-1]!="?"):
           endOfQuestions=list(vector).index(i)
           break
    endOfDecision=len(vector)
    return [endOfQuestions,endOfDecision]
list_of_end_index_quest_dec=GetProperties(myquestions)

decision_tree = (DecisionTree(new_df,list_of_end_index_quest_dec[0],list_of_end_index_quest_dec[0],list_of_end_index_quest_dec[1],myquestions))
secondlist = BinaryTreePrinter(decision_tree)
print(DecisionPredictor(decision_tree, list(df.iloc[:,319]), myquestions, secondlist))
end_time = time.time()
print(f"Czas pracy aplikacji: {end_time - start_time} sekund.")
