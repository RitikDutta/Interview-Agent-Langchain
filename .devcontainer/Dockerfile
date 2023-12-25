# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Update OS packages
RUN apt-get update && apt-get upgrade -y

# Install any needed OS packages here
RUN apt-get install -y git libpq-dev

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]

