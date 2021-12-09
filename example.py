import numpy as np
import matplotlib.pyplot as plt
#import SimplicialComplex
import itertools
import functools
from ripser import ripser
from persim import plot_diagrams


import numpy as np
import matplotlib.pyplot as plt

#----------------------------------Example code---------------------------------

# n = 30 #number of points to generate

# #generate space of parameter
# theta = np.linspace(0, 2.0*np.pi, n) 

# a, b, r = 0.0, 0.0, 5.0

# x = a + r*np.cos(theta)
# y = b + r*np.sin(theta)

# #code to plot the circle for visualization
# plt.plot(x, y)
# plt.show()

# x2 = np.random.uniform(-0.75,0.75,n) + x #add some "jitteriness" to the points
# y2 = np.random.uniform(-0.75,0.75,n) + y

# print('-------------------x2---------------------')
# print(x2)
# fig, ax = plt.subplots()
# ax.scatter(x2,y2)
# plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def euclidianDist(a,b): #this is the default metric we use but you can use whatever distance function you want
    return np.linalg.norm(a - b) #euclidian distance metric


#Build neighorbood graph
def buildGraph(raw_data, epsilon = 3.1, metric=euclidianDist): #raw_data is a numpy array
    nodes = [x for x in range(raw_data.shape[0])] #initialize node set, reference indices from original data array
    edges = [] #initialize empty edge array
    weights = [] #initialize weight array, stores the weight (which in this case is the distance) for each edge
    for i in range(raw_data.shape[0]): #iterate through each data point
        for j in range(raw_data.shape[0]-i): #inner loop to calculate pairwise point distances
            a = raw_data[i]
            b = raw_data[j+i] #each simplex is a set (no order), hence [0,1] = [1,0]; so only store one
            if (i != j+i):
                dist = metric(a,b)
                if dist <= epsilon:
                    edges.append({i,j+i}) #add edge if distance between points is < epsilon
                    weights.append(dist)
    return nodes,edges,weights

def lower_nbrs(nodeSet, edgeSet, node): #lowest neighbors based on arbitrary ordering of simplices
    return {x for x in nodeSet if {x,node} in edgeSet and node > x}

def ripsFiltration(graph, k): #k is the maximal dimension we want to compute (minimum is 1, edges)
    nodes, edges, weights = graph
    VRcomplex = [{n} for n in nodes]
    filter_values = [0 for j in VRcomplex] #vertices have filter value of 0
    for i in range(len(edges)): #add 1-simplices (edges) and associated filter values
        VRcomplex.append(edges[i])
        filter_values.append(weights[i])
    if k > 1:
        for i in range(k):
            for simplex in [x for x in VRcomplex if len(x)==i+2]: #skip 0-simplices and 1-simplices
                #for each u in simplex
                nbrs = set.intersection(*[lower_nbrs(nodes, edges, z) for z in simplex])
                for nbr in nbrs:
                    newSimplex = set.union(simplex,{nbr})
                    VRcomplex.append(newSimplex)
                    filter_values.append(getFilterValue(newSimplex, VRcomplex, filter_values))

    return sortComplex(VRcomplex, filter_values) #sort simplices according to filter values

def getFilterValue(simplex, edges, weights): #filter value is the maximum weight of an edge in the simplex
    oneSimplices = list(itertools.combinations(simplex, 2)) #get set of 1-simplices in the simplex
    max_weight = 0
    for oneSimplex in oneSimplices:
        filter_value = weights[edges.index(set(oneSimplex))]
        if filter_value > max_weight: max_weight = filter_value
    return max_weight


def compare(item1, item2): 
    #comparison function that will provide the basis for our total order on the simpices
    #each item represents a simplex, bundled as a list [simplex, filter value] e.g. [{0,1}, 4]
    if len(item1[0]) == len(item2[0]):
        if item1[1] == item2[1]: #if both items have same filter value
            if sum(item1[0]) > sum(item2[0]):
                return 1
            else:
                return -1
        else:
            if item1[1] > item2[1]:
                return 1
            else:
                return -1
    else:
        if len(item1[0]) > len(item2[0]):
            return 1
        else:
            return -1

def sortComplex(filterComplex, filterValues): #need simplices in filtration have a total order
    #sort simplices in filtration by filter values
    pairedList = zip(filterComplex, filterValues)
    #since I'm using Python 3.5+, no longer supports custom compare, need conversion helper function..its ok
    sortedComplex = sorted(pairedList, key=functools.cmp_to_key(compare)) 
    sortedComplex = [list(t) for t in zip(*sortedComplex)]
    #then sort >= 1 simplices in each chain group by the arbitrary total order on the vertices
    orderValues = [x for x in range(len(filterComplex))]
    return sortedComplex

def drawComplex(origData, ripsComplex, axes=[-6,8,-6,6]):
  plt.clf()
  plt.axis(axes)
  plt.scatter(origData[:,0],origData[:,1]) #plotting just for clarity
  for i, txt in enumerate(origData):
      plt.annotate(i, (origData[i][0]+0.05, origData[i][1])) #add labels

  #add lines for edges
  for edge in [e for e in ripsComplex if len(e)==2]:
      #print(edge)
      pt1,pt2 = [origData[pt] for pt in [n for n in edge]]
      #plt.gca().add_line(plt.Line2D(pt1,pt2))
      line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor='r')
      plt.gca().add_line(line)

  #add triangles
  for triangle in [t for t in ripsComplex if len(t)==3]:
      pt1,pt2,pt3 = [origData[pt] for pt in [n for n in triangle]]
      line = plt.Polygon([pt1,pt2,pt3], closed=False, color="blue",alpha=0.3, fill=True, edgecolor=None)
      plt.gca().add_line(line)
  plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import gzip
import matplotlib.pyplot as plt
import numpy as np

#---------------
import cv2

f = gzip.open('train-images-idx3-ubyte.gz','r')
image_size = 28 #datset of 60,000 small square 28x28 pixel greyscale images
num_images = 60000
f.read(16)
buf = f.read(image_size * image_size * num_images)
print(len(buf))
data = np.frombuffer(buf, dtype=np.uint8).astype(np.int32) #interpret a buffer as a one dimensioanl array where the array is a float number
# print('-----------------------------------Data is:-------------------------')
# print(data)
# print(len(data))

data = data.reshape(num_images, image_size, image_size, 1)  ##reshape the one dimensional vector to 28 vectors of a length of 28 (each)
# print('---------------------------------------END READING IMAGE-------------------------------------------------')
# print(data)

image = np.asarray(data[59999]).squeeze()#<------choose the image from MNIST dataset, now is reading the image 59991 from the 60,000 images it has
#print(image)
plt.imshow(image)
plt.show()



sample = np.array([])

for array in image:
    sample = np.concatenate([sample, array])
#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
#np.concatenate(image)
image=np.flip(image)
print('---------------------Imagen concatenada---------------------')
print(image)
print(sample)
yx_coords = np.column_stack(np.where(sample>= 0))
xy_coords = np.flip(np.column_stack(np.where(image > 0)), axis=1)
np.concatenate(xy_coords)
print(xy_coords)


#--------------------------------------------------------Data that we can use--------------------------------------------------------------
# data = np.array(list(zip(x2,y2)))
# data = np.array([[1,4],[1,1],[6,1],[6,4]])
# print('------------------------------------------------------------')
# print(data)
# data =np.array([[8,3],[9,3],[10,3],[8,4],[9,4],[10,4],[8,5],[9,5],[10,5],[8,6],[9,6],[10,6],[8,7],[9,7],[10,7],[8,8],[9,8],[10,8]])
data=np.array(np.flip(np.flip(xy_coords)))   #-------------------data of a number of MNIST dataset-----------

#------------------Calculating homology using ripser library------------------------------
#----------------------------------------Ripser--------------------------------------------
diagrams = ripser(data,maxdim=2)['dgms']
print('---------------Diagrams----------------------')
print(diagrams)
plot_diagrams(diagrams, show=True)
graph2 = buildGraph(raw_data=data, epsilon=1.5) #<------------change the value of epsilon to see the simplicial complex
ripsComplex2 = ripsFiltration(graph2, k=3) #k simplex
drawComplex(origData=data, ripsComplex=ripsComplex2[0], axes=[-5,30,-5,25])
print('----END------')

# bm = filterBoundaryMatrix(ripsComplex2)
# bm #Here is the (non-reduced) boundary matrix
# z = reduceBoundaryMatrix(bm)
