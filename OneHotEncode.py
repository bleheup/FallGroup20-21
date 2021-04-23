
import numpy as np

def OneHotEncode(observation):
    newObservation = np.zeros(249, dtype=int)
    for i in range(45):
        newObservation[i] = observation[i]

    iterationCounter = 0
    #newObservation[45+observation[45]] = 1
    for j in range(45, 105, 5):

        #translate what node location
        LocationIndex = int(j + 12*iterationCounter + observation[j])
        newObservation[LocationIndex] = 1

        #translate what type of group
        GroupTypeindex = int(j+ 12*iterationCounter + 11 + observation[j+1])
        newObservation[GroupTypeindex] = 1


        #translate average health
        AverageHealthIndex = int(j+ 12*iterationCounter + 14)
        newObservation[AverageHealthIndex] = observation[j+2]

        #translate transit status
        TransitStatusIndex = int(j+ 12*iterationCounter + 15)
        newObservation[TransitStatusIndex] = observation[j+3]

        #translate number of units alive
        NumberofUnitsAliveIndex = int(j+ 12*iterationCounter + 16)
        newObservation[NumberofUnitsAliveIndex] = observation[j+4]
        iterationCounter += 1
    
    return newObservation


#Testing data
observation = [  64.,    0.,    0.,  500.,    0.,    0.,    1.,  100.,    8.,
          0.,    0.,  100.,    0.,    1.,    0.,  100.,    0.,    0.,
          0.,  -88.,    0.,    0.,    0., -100.,   16.,    0.,    0.,
        100.,   12.,    0.,    1., -100.,   16.,    0.,    0.,  -50.,
          7.,    1.,    0.,  -13.,    8.,    0.,    0., -500.,   32.,
          3.,    1.,   94.,    1.,    8.,   10.,    2.,   56.,    1.,
          7.,    8.,    0.,   93.,    0.,    8.,    6.,    1.,   58.,
          0.,    8.,    8.,    2.,    0.,    0.,    0.,    4.,    0.,
        100.,    0.,    8.,    7.,    1.,   90.,    0.,    8.,    2.,
          2.,  100.,    0.,    8.,    7.,    0.,  100.,    0.,    8.,
          2.,    1.,   15.,    0.,    7.,    9.,    2.,   70.,    0.,
          8.,    2.,    0.,   91.,    1.,   12.]

observation2 = [  99.,    0.,    0.,  500.,    0.,    0.,    1.,   53.,   13.,
          0.,    0.,    4.,   12.,    1.,    0.,  100.,    0.,    0.,
          0., -100.,    0.,    0.,    0.,  -28.,    8.,    0.,    0.,
        100.,    0.,    0.,    1.,  -44.,    0.,    0.,    0.,  -92.,
          8.,    1.,    0., -100.,   16.,    0.,    0., -500.,   24.,
          9.,    1.,   83.,    1.,    8.,    4.,    2.,    9.,    0.,
          1.,   11.,    0.,    0.,    0.,    0.,   10.,    1.,    0.,
          0.,    0.,    8.,    2.,    0.,    0.,    0.,    2.,    0.,
         87.,    0.,    8.,    7.,    1.,   90.,    0.,    8.,    9.,
          2.,   82.,    1.,    8.,    8.,    0.,   90.,    0.,    8.,
          1.,    1.,   12.,    0.,    5.,    6.,    2.,    0.,    0.,
          0.,    4.,    0.,   85.,    0.,   12.]

print(OneHotEncode(observation))
