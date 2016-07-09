import numpy as np
from base.rls import RLS

class ePLRegressor(object):

	def __init__(self, alpha=0.01, beta=0.12, lambd=0.84, tau=0.10):
		self.a = 0.00
		self.params = {
			'alpha' : alpha, 'beta' : beta, 'tau' : tau, 'lambd' : lambd
		}


	def compat(self, sample, cluster):
		return 1.00 - np.linalg.norm(sample - cluster)/len(sample)


	def arousal(self, a, p_max):
		return a + self.params['beta'] * (1 - p_max - a)


	def gaussian(self, sample, cluster):
		return np.exp(-np.power(np.linalg.norm(cluster - sample), 2)/0.09)


	def wipe(self, mi):
		for i in range(len(self.clusters) - 1):
			l = np.array([])
			if i < len(self.clusters) - 1:
				for j in range(i + 1, len(self.clusters)):
					vi, vj = self.clusters[i], self.clusters[j]
					if self.compat(vi['pos'], vj['pos']) >= self.params['lambd']:
						vi['pos'] = (vi['pos'] + vj['pos'])/2,
						vi['coefs'].setw(vi['coefs'].getw() + vj['coefs'].getw()/2)

						mi[i] = (mi[i] + mi[j])/2 
						l = np.append(l, int(j))

				self.clusters = [i for j, i in enumerate(self.clusters) if j not in l]
				mi = np.delete(mi, (l))


	def _build_coefs(self, x_ext, y, mi):
		rl = sum(i*j['coefs'].getw() for i, j in zip(mi, self.clusters))
		return RLS(w=rl, gama=0.9)


	def evolve(self, x, y = 0.00):

		x_ext = np.append(1., x)
		
		# Checking for system prior knowledge
		if not hasattr(self, 'know'):
			self.know = True
			# Setting the sample as the first cluster
			self.clusters = [{
				'pos': x_ext,
				'coefs': RLS(w=np.zeros((1, len(x_ext))), gama=0.9)
			}]
			return y

		# Compatibility measure
		p = np.array([self.compat(x_ext, c['pos']) for c in self.clusters])
		# Arousal index
		self.a = self.arousal(self.a, max(p))
		# Degree of activation
		mi = np.array([self.gaussian(x_ext, c['pos']) for c in self.clusters])

		if self.a > self.params['tau']:
			# Creating a new cluster
			self.clusters.append({
				'pos': x_ext,
				'coefs': self._build_coefs(x_ext, y, mi)
			})

			mi = np.append(mi, 1.00)
		else:
			s = self.clusters[np.argmax(p)]['pos']

			# Updating cluster value
			self.clusters[np.argmax(p)] = {
				'pos': s + self.params['alpha'] * (np.power(max(p), 1.00 - self.a)) * (x_ext - s),
				'coefs': self.clusters[np.argmax(p)]['coefs'].update(y=y, X=x_ext)
			}

			# Removing redundant clusters
			if len(self.clusters) > 1:
				self.wipe(mi)

		mi = mi / sum(mi)
		''' Model output: {y} is {float} in [-inf, inf] '''
		y2 = sum([np.dot(np.array(i * j['coefs'].getw()), x_ext) for i, j in zip(mi, self.clusters)])
		return y2