import numpy as np
from base.krls import KRLS
import subprocess


class ePLKRLSRegressor(object):

	def __init__(self, alpha=0.01, beta=0.12, lambd=0.85, tau=0.15):
		self.a = 0.00
		self.params = {
			'alpha' : alpha, 'beta' : beta, 'tau' : tau, 'lambd' : lambd
		}
		self.c = 1

	def compat(self, sample, cluster):
		return 1.00 - np.linalg.norm(sample[1:] - cluster[1:])/len(sample[1:])

	def plot_centers(self, data = None):
		#print np.array([np.array(i['pos']) for i in self.clusters])
		np.savetxt('centers.txt', np.array([i['pos'] for i in self.clusters]))
		np.savetxt('real.txt', data)
		proc = subprocess.Popen(['gnuplot','-p'],
		                        shell=True,
		                        stdin=subprocess.PIPE,
		                        )

		proc.stdin.write("plot 'centers.txt' using 2:3 with points notitle, 'real.txt' using 1:2 with points notitle\n")
		proc.stdin.write('pause 1000\n')
		proc.stdin.write('quit\n')

	def arousal(self, a, p_max):
		return a + self.params['beta'] * (1 - p_max - a)

	def gaussian(self, sample, cluster):
		return np.exp(-np.power(np.linalg.norm(cluster - sample), 2)/0.09)

	def wipe(self):

		for i in range(len(self.clusters) - 1):
			l = np.array([])
			if i < len(self.clusters) - 1:
				for j in range(i + 1, len(self.clusters)):
					vi, vj = self.clusters[i], self.clusters[j]

					if self.compat(vi['pos'], vj['pos']) >= self.params['lambd']:
						vi['pos'] = (vi['pos'] + vj['pos'])/2
						l = np.append(l, int(j))
				self.clusters = [i for j, i in enumerate(self.clusters) if j not in l]

	def _build_coefs(self, x_ext, y):
		krls = KRLS(params=dict(adopt_thresh=0.01, dico_max_size=100))
		krls.update(x_ext, y)
		return krls

	def evolve(self, x, y = 0.00):
		x_ext = np.append(1., x)

		# Checking for system prior knowledge
		if not hasattr(self, 'know'):
			self.know = True
			# Setting the sample as the first cluster

			self.clusters = [{
				'pos': x_ext,
				'coefs': self._build_coefs(x_ext, y)
			}]
			return y

		# Compatibility measure
		p = np.array([self.compat(x_ext, c['pos']) for c in self.clusters])
		# Arousal index
		self.a = self.arousal(self.a, max(p))

		if self.a > self.params['tau']:
			# Creating a new cluster
			self.clusters.append({
				'pos': x_ext,
				'coefs': self._build_coefs(x_ext, y)
			})
		else:
			s = self.clusters[np.argmax(p)]['pos']

			# Updating cluster value
			self.clusters[np.argmax(p)]['coefs'].update(x_ext, y)
			self.clusters[np.argmax(p)] = {
				'pos': s + self.params['alpha'] * (np.power(max(p), 1.00 - self.a)) * (x_ext - s),
				'coefs': self.clusters[np.argmax(p)]['coefs']
			}

			# Removing redundant clusters
			if len(self.clusters) > 1:
				self.wipe()

		# Degree of activation
		mi = np.array([self.gaussian(x_ext, c['pos']) for c in self.clusters])
		mi = mi / sum(mi)
		''' Model output: {y} is {float} in [-inf, inf] '''

		# Eq. 35
		self.c = self.c + 1
		y2 = sum([float(j['coefs'].query(x_ext)) * i for i, j in zip(mi, self.clusters)])

		return y2
