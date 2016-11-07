class LogisticRegression:
    def __init__(self, learning_rate, epsilon):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def generate_line(self):
        x1, x2, y1, y2 = np.random.uniform(-1, 1, 4)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y2 - slope * x1
        self.line = np.array([-intercept, -slope, 1])
        
    def generate_points(self, n):
        return np.array([(1, x, y) for x, y in zip(np.random.uniform(-1, 1, n), 
                                                          np.random.uniform(-1, 1, n))])

    def cal_signs(self, points, line):
        def sign(x):
            if abs(x) < 0.000000001:
                return 0
            return x > 0 and 1 or -1
        signv = np.vectorize(sign)
        signs = signv(points.dot(line))
        return signs
    
    def train(self, n, plot=False):
        self.points = self.generate_points(n)
        self.generate_line()
        self.signs = self.cal_signs(self.points, self.line)
        self.w = np.array([0.0, 0.0, 0.0])
        iterations = 0
        while True:
            iterations += 1
            indices = np.random.permutation(np.arange(n))
            prev_w = np.copy(self.w)
            for index in indices:
                point = self.points[index]
                sign = self.signs[index]
                delta = -self.learning_rate * sign * point / (1 + np.exp(sign * self.w.dot(point)))
                self.w -= delta
                
            if np.sqrt((prev_w - self.w).dot(prev_w - self.w)) < self.epsilon:
                break

        if plot:
            plt.scatter(self.points[:, 1], self.points[:, 2])
            x = np.array([-1, 1])
            train_y = -self.w[1]/self.w[2] * x - self.w[0] / self.w[2]
            plt.plot(x, train_y, 'k-')
            y = -x * self.line[1] - self.line[0]
            plt.plot(x, y, 'r-')
            plt.show()
            
        return self.w, iterations
    
    def train_error(self):
        train_signs = self.cal_signs(self.points, self.w)
        return sum(train_signs != self.signs) / len(self.points)
        
    def test_error(self, n_test, plot=False):
        test_points = self.generate_points(n_test)
        if plot:
            plt.scatter(test_points[:, 1], test_points[:, 2])
            x = np.array([-1, 1])
            train_y = -self.w[1]/self.w[2] * x - self.w[0] / self.w[2]
            plt.plot(x, train_y, 'k-')
            y = -x * self.line[1] - self.line[0]
            plt.plot(x, y, 'r-')
            plt.show()
        signs = self.cal_signs(test_points, self.line)
        error = np.log(1.0 + np.exp(-signs * test_points.dot(self.w))).mean()
        return error



total_test_error = 0
total_iterations = 0
for i in range(100):
    print i
    lr = LogisticRegression(0.01, 0.01)
    w, iterations = lr.train(100)
    total_test_error += lr.test_error(100)
    total_iterations += iterations
    
print total_test_error / 100
print total_iterations / 100


def cal_error(u, v):
    return (u*np.exp(v)-2*v*np.exp(-u))**2
# gradient descent
i = 0
u = 1
v = 1
learning_rate = 0.1
while cal_error(u, v) >= 1e-14:
    delta_u = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v)+2*v*np.exp(-u))
    delta_v = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v)-2*np.exp(-u))
    u -= learning_rate * delta_u
    v -= learning_rate * delta_v
    i += 1

print u, v

# coordinate descent
i = 0
u = 1
v = 1
learning_rate = 0.1
while i < 15:
    delta_u = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(np.exp(v)+2*v*np.exp(-u))
    u -= learning_rate * delta_u
    delta_v = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v)-2*np.exp(-u))
    v -= learning_rate * delta_v
    i += 1

cal_error(u, v)