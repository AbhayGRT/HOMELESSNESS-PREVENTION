# Hacksparrow - A SOLUTION TO TRACK AND HELP HOMELESS PEOPLE

![Goal](https://img.shields.io/badge/Goal-Social_Welfare-blue)
![focus](https://img.shields.io/badge/Tech-Web_Dev-brightgreen)
![focus](https://img.shields.io/badge/Tech-ML-brightgreen)

## PROJECT DESCRIPTION

### APPROACH

Slums are generally characterized by extremely high population densities, irregular arrangement of small buildings, and poor living conditions. These areas often harbor homeless populations. Our approach involves:

1. **Identification of Slum Areas**: We identify areas with extensive slum presence, such as Dharavi, one of the largest slums. This is done through surveys and user inputs.

2. **Location Mapping**: Collected data is marked on maps to track slum areas.

3. **Machine Learning Integration**: Using satellite imagery (Sentinel) and machine learning, we train models to predict probable slum locations based on collected data.

4. **Validation and Tracking**: Probable locations are validated against user inputs and surveys. This helps in tracking homeless populations effectively.

5. **Additional Features**: We have incorporated features such as:
   - User-input location storage in a database.
   - Contacting nearby NGOs.
   - Donation options via QR codes.
   - Reporting suspicious activities like kidnapping, smuggling, or rape cases.
   - Organizing and attending events like free health checkups, treatments, food distribution, and educational sessions.

## FILE DESCRIPTION

- **Mumbai_details.html**: Details of slums or homeless populations displayed using various maps (Google map, Satellite map, and Machine Learning classified map).
- **Support-us.html**: Allows users to donate for homeless people using QR codes.
- **contactauthorities.html**: Allows users to contact managing authorities.
- **contactNGO.html**: Allows users to contact nearby NGOs.
- **index.html**: Home page.
- **contribute.html**: Allows users to contribute the location of homeless people stored in a database.
- **file_report.html**: Allows users to report any deformities or suspicious activities related to homeless people shown on the map.
- **index-logout.html**: Allows users to logout from the website.
- **Login.css**: Stylesheet for the login page.
- **login.html**: Login page.
- **mission.html**: Describes the mission of the project.
- **signup2.css**: Stylesheet for the signup page.
- **signup2.html**: Signup page.
- **package.json**: Stores important metadata about the project.
- **feature_collection.js**: Dataset for ML classified map to be imported in Google Earth Engine.
- **ml_map_gee.js**: Machine learning code used in Google Earth Engine editor to predict probable slum areas.
- **github_repo_turing_crypt.txt**: GitHub repository link.


# SETTING UP THE PROJECT

## Requirements

For development, you will only need Node.js and a node global package, Yarn, installed in your environement.

### Node

- #### Node installation on Windows

  Just go on [official Node.js website](https://nodejs.org/) and download the installer.
  Also, be sure to have `git` available in your PATH, `npm` might need it (You can find git [here](https://git-scm.com/)).

- #### Node installation on Ubuntu

  You can install node.js and npm easily with apt install, just run the following commands.

      $ sudo apt install nodejs
      $ sudo apt install npm

- #### Other Operating Systems
  You can find more information about the installation on the [official Node.js website](https://nodejs.org/) and the [official NPM website](https://npmjs.org/).

If the installation was successful, you should be able to run the following command.

    $ node --version
    v8.11.3

    $ npm --version
    6.1.0

If you need to update `npm`, you can make it using `npm`! Cool right? After running the following command, just open again the command line and be happy.

    $ npm install npm -g

## Authors

- [@Jayesh Pandey](https://github.com/jayesh-RN)
- [@Arpitshivam Pandey](https://github.com/Arpit0324)
- [@Abhay Maurya](https://github.com/AbhayGRT)

## Contributing

Contributions are always welcome!

## Tech Stack

**Client:** HTML, CSS, JAVASCRIPT

**Server:** NodeJS, MongoDB

**MACHINE LEARNING:** Pre-processing, Multiclass Classification using Random Forest, NLTK, PyTorch, ReLU

**Dashboard:** PowerBI
