class LinearRegression:
    def __init__(self, non_linear=False):
        self.non_linear = non_linear
        
    def generate_line(self, non_linear=False):
        if self.non_linear:
            self.line = np.array([-0.6, 0, 0, 0, 1, 1])
        else:
            x1, x2, y1, y2 = np.random.uniform(-1, 1, 4)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y2 - slope * x1
            self.line = np.array([-intercept, -slope, 1])
        
    def generate_points(self, n):
        if self.non_linear:
            return np.array([(1, x, y, x*y, x**2, y**2) for x, y in zip(np.random.uniform(-1, 1, n), 
                                                          np.random.uniform(-1, 1, n))])
        else:
            return np.array([(1, x, y) for x, y in zip(np.random.uniform(-1, 1, n), 
                                                          np.random.uniform(-1, 1, n))])

    def cal_signs(self, points, line):
        def sign(x):
            if abs(x) < 0.000000001:
                return 0
            return x > 0 and 1 or -1
        signv = np.vectorize(sign)
        signs = signv(points.dot(line))
        if self.non_linear:
            n_points = len(signs)
            indices = np.random.randint(low=0, high=n_points, size=int(0.1*n_points))
            signs[indices] = -1 * signs[indices] 
 
        return signs
    
    def train(self, n, transform_points=False, plot=False):
        self.points = self.generate_points(n)
        self.generate_line()
        self.signs = self.cal_signs(self.points, self.line)
        if (self.non_linear) & (not transform_points):
            self.points = self.points[:, :3]
        self.w = np.linalg.inv(self.points.T.dot(self.points)).dot(self.points.T).dot(self.signs)
#         print self.w
        if plot:
            plt.scatter(self.points[:, 1], self.points[:, 2])
            x = np.array([-1, 1])
            train_y = -self.w[1]/self.w[2] * x - self.w[0] / self.w[2]
            plt.plot(x, train_y, 'k-')
            y = -x * self.line[1] - self.line[0]
            plt.plot(x, y, 'r-')
            plt.show()
            
        return self.w
    
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
        test_signs = self.cal_signs(test_points, self.w)
        return sum(test_signs != signs) / n_test
    
    def pla(self, n):
        self.train(n)
        train_line = np.copy(self.w)
        n_iterations = 0
        misclassified_points = np.copy(self.points[self.cal_signs(self.points, train_line) != self.signs])
#         print len(misclassified_points)
        while len(misclassified_points) != 0:
    #         plt.scatter(points[:, 1], points[:, 2])
    #         x = np.array([-1, 1])
    #         train_y = -train_line[1]/train_line[2] * x - train_line[0] / train_line[2]
    #         plt.plot(x, train_y, 'k-')
    #         y = -x * line[1] - line[0]
    #         plt.plot(x, y, 'r-')
    #         plt.show()
    
            selected_point = misclassified_points[np.random.randint(low=0, high=len(misclassified_points), size=1)][0]
            selected_sign = self.cal_signs(selected_point, self.line)
#             print selected_point.dot(train_line)
            train_line += selected_point * selected_sign
            train_signs = cal_signs(self.points, train_line)
            misclassified_points = self.points[train_signs != self.signs]
            n_iterations += 1
        
        return n_iterations

# 5
total_error = 0
for i in range(1000):
    lr = LinearRegression()
    lr.train(100)
    total_error += lr.train_error()
    
print total_error / 1000

# 6
total_test_error = 0
for i in range(1000):
    lr = LinearRegression()
    lr.train(100)
    total_test_error += lr.test_error(1000)
    
print total_test_error / 1000

# 7
iterations = 0
for i in range(1000):
    lr = LinearRegression()
    iterations += lr.pla(10)

print iterations / 1000

# 8
total_error = 0
for i in range(1000):
    lr = LinearRegression(non_linear=True)
    lr.train(1000, transform_points=False)
    total_error += lr.train_error()
    
print total_error / 1000

# 9, 10
total_error = 0
train_line = np.zeros((1000, 6))
for i in range(1000):
    lr = LinearRegression(non_linear=True)
    train_line[i] = lr.train(1000, transform_points=True)
    total_error += lr.train_error()
    
print train_line.mean(axis=0)
print total_error / 1000