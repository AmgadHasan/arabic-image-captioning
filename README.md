# Arabic Image Captionig
### A encoder-decoder deep learning based model to detect Arabic dialects deployed as a fastapi.
Request
Response
## Features
1. Raw Arabic captions dataset
2. preprocessed train and test sets
3. Deep learning model with attention (CNN encoder + RNN Decoder)
4. fastapi server

## Getting Started
To get started with the project, you will need to clone the repository to your local machine:<br>
`git clone git@github.com:AmgadHasan/arabic-image-captioning.git`
<br><br>Once you have cloned the repository, you can open the project in your preferred code editor and start exploring the code.

## Prerequisites
To run the project, you will need to have the following installed on your machine:
1. tensorflow (for the deep learning model)
2. tensorflow, fastapi, uvicron, pyantic (for the api)

You can run the following command to install these packages:<br>
`pip install -r requirements.txt`
## Running the project
To run the api server, go to the code directory and run the following command:<br>
`de utils; python -m uvicorn api:app --reload`
## Contributing
Contributions are welcome! To contribute, please follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix: git checkout -b my-new-feature.

Make changes and commit them: <br>
`git commit -am 'Add some feature'`<br>

Push to the branch: <br>
`git push origin my-new-feature`

Submit a pull request.

## License
This project is licensed under the [Apache 2.0](https://github.com/AmgadHasan/arabic-dialect-detection/blob/main/LICENSE) License - see the LICENSE file for details.

## Authors
This project was created by:
1. [Amgad Hasan](https://github.com/AmgadHasan)
2. [Abdelwahab Elghandour](https://github.com/Elghandour-eng)


## Acknowledgments

## Copyritghs
   Copyright 2023 Amgad Hasan, Abdelwaha Elghandour

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
