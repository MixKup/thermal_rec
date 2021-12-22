# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=35, maxDistt = 300):

		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDistt = maxDistt
		self.maxDisappeared = maxDisappeared
		self.face = OrderedDict()
		self.flag = OrderedDict()

	def register(self, centroid, fbox):

		self.objects[self.nextObjectID] = centroid
		self.flag[self.nextObjectID] = 1
		self.face[self.nextObjectID] = fbox
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):

		del self.objects[objectID]
		del self.face[objectID]
		del self.flag[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		flag = 0
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)


			return self.objects, self.face, self.flag
		for objectID in list(self.disappeared.keys()):
			self.flag[objectID] = 0
		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], rects[i])
				flag = 1

		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			#D = D * 1/((D < self.maxDistt)+10**(-8))

			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				
				objectID = objectIDs[row]
				#FIXME:
				# if dist.cdist(self.objects[objectID], inputCentroids[col]) > self.maxDistt: 
					# continue 

				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				self.face[objectID] = rects[col]

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			for col in unusedCols:
				self.register(inputCentroids[col], rects[col])
				flag = 1

		return self.objects, self.face, self.flag
