FROM python:3.7.3-slim

# Install jupyter:
RUN pip3 install jupyter

# update package lists (to be able to install sudo, nano and othe things needed for linux)
RUN apt-get update
RUN apt-get install nano
RUN apt-get install sudo

# alternatively, you can list all the dependencies in a separate file:
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# expose a port to the host machine:
EXPOSE 8888

# Start the jupyter notebook
# ENTRYPOINT ["jupyter", "notebook", "--ip=*", "--allow-root"]