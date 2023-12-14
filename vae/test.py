import matplotlib.pyplot as plt

# %%
digit = 1
images = []
index = 1
for x, y in dataset:
    if y == index:
        images.append(x)
        index += 1
    if index == 3:
        break


# %%
images
# %%

mu, sigma = model.encode(images[0].view(1, INPUT_DIMENSIONS))
mu1, sigma1 = model.encode(images[1].view(1, INPUT_DIMENSIONS))
print(f"mu: {mu}, sigma: {sigma}")
# %%
epsilon = torch.randn_like(sigma)
# %%
z = mu + epsilon * sigma
# %%
out = model.decode(z)
out = out.view(-1, 1, 28, 28)
save_image(out, f"output/asd_02.png")
# %%


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Set mean and standard deviation
mean = mu[0][0].item()
std_dev = np.sqrt(sigma[0][0].item()**2)

# Generate data points for the x-axis
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)

# Calculate the corresponding probabilities for the y-axis using the normal distribution
y = norm.pdf(x, mean, std_dev)

# Plot the Gaussian distribution
plt.figure(figsize=(8, 6))
plt.plot(x, y, color='blue', label=f'Gaussian Distribution\nMean={mean}, Std Dev={std_dev}')
plt.title('Gaussian Distribution')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()




# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
number_of_graphs = 20

# Assuming mu and sigma are tensors or arrays with shape (1, 20)
mu_values = mu[0][:number_of_graphs].detach().numpy()  # Extract first 8 means
sigma_values = np.sqrt(sigma[0][:number_of_graphs].detach().numpy() ** 2)  # Extract first 8 standard deviations

plt.figure(figsize=(8, 6))

# Plot individual Gaussian distributions
colors = plt.cm.tab10(np.linspace(0, 1, number_of_graphs))
for i in range(number_of_graphs):
    mean = mu_values[i].item()
    std_dev = sigma_values[i].item()
    
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    
    plt.plot(x, y, label=f'Distribution {i+1}', color=colors[i])

plt.title('Gaussian Distributions')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.legend(loc='upper right')  # Move legend to upper right corner
plt.grid(True)
plt.show()




# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
number_of_graphs = 20

# Assuming mu and sigma are tensors or arrays with shape (1, 20)
new_sigma = sigma * epsilon
mu_values = mu[0][:number_of_graphs].detach().numpy()  # Extract first 8 means
sigma_values = np.sqrt(new_sigma[0][:number_of_graphs].detach().numpy() ** 2)  # Extract first 8 standard deviations

plt.figure(figsize=(8, 6))

# Plot individual Gaussian distributions
colors = plt.cm.tab10(np.linspace(0, 1, number_of_graphs))
for i in range(number_of_graphs):
    mean = mu_values[i].item()
    std_dev = sigma_values[i].item()
    
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    
    plt.plot(x, y, label=f'Distribution {i+1}', color=colors[i])

plt.title('Gaussian Distributions')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.legend(loc='upper right')  # Move legend to upper right corner
plt.grid(True)
plt.show()




# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
number_of_graphs = 20

# Assuming mu and sigma are tensors or arrays with shape (1, 20)
new_sigma = epsilon
mu_values = mu[0][:number_of_graphs].detach().numpy()  # Extract first 8 means
sigma_values = np.sqrt(new_sigma[0][:number_of_graphs].detach().numpy() ** 2)  # Extract first 8 standard deviations

plt.figure(figsize=(8, 6))

# Plot individual Gaussian distributions
colors = plt.cm.tab10(np.linspace(0, 1, number_of_graphs))
for i in range(number_of_graphs):
    mean = mu_values[i].item()
    std_dev = sigma_values[i].item()
    
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    
    plt.plot(x, y, label=f'Distribution {i+1}', color=colors[i])

plt.title('Gaussian Distributions')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.legend(loc='upper right')  # Move legend to upper right corner
plt.grid(True)
plt.show()

# %%
z2 = mu + epsilon
img2 = model.decode(z2)
out = img2.view(-1, 1, 28, 28)
save_image(out, f"output/img3.png")



# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Assuming z, mu, and sigma are 1x20 vectors
# z, mu, and sigma are already defined in your context

# Calculate mean and covariance matrix of the Gaussian distribution
mean = mu.mean()
cov_matrix = np.cov(mu, rowvar=False)

# Generate grid points for the Gaussian distribution
x, y = np.meshgrid(np.linspace(min(mu) - 1, max(mu) + 1, 100), np.linspace(min(sigma) - 1, max(sigma) + 1, 100))
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov_matrix)

# Visualize the Gaussian distribution as a contour plot
plt.figure(figsize=(8, 6))
plt.contourf(x, y, rv.pdf(pos), cmap='Blues', levels=30)
plt.colorbar(label='Probability Density')
plt.xlabel('mu')
plt.ylabel('sigma')
plt.title('Gaussian Distribution')
plt.grid(True)

# Plot the point z on the distribution
plt.scatter(mu, sigma, color='red', label='Point z', zorder=5)
plt.legend()
plt.show()

# %%
