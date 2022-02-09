FROM tiangolo/node-frontend:10

WORKDIR /app
COPY package.json ./
COPY package-lock.json ./
RUN npm install --silent
RUN npm install @chakra-ui/core @emotion/core @emotion/styled emotion-theming --silent
COPY . ./
EXPOSE 3000
CMD ["npm", "start"]