# Robotteknologi-3.-semester
In this repository you find our solution for a system that takes input images from underwater enviorments, and enhance them to remove noise and correct colours. This repository consist of our solution and data collected from a test enviorment, there are also some of our results both as output images and image validation metric results.

## Folder Structure
Downunder you can see the repository structure for making it easier to navigate

```
P3/
├── ColorCorrection/
├── Dehazing/
├── Input/
├── Modules/                    
├── Objective_testing/
├── Palette_detection/
├── Results/
│   ├── Data/             #Collected data and results from this data
│   │   ├── Clay/
│   │   ├── Gips/
│   │   ├── Milk/
│   │   ├── Spinat/
│   │   └── Water/
├── __pycache__/
```

### Key Folders:

- **ColorCorrection**: Program for applying colour correction.
- **Dehazing**: Program for using dehazing on images.
- **Input**: Integration with camera.
- **Modules**: Core functionality modules.
- **Objective_testing**: Program for applying image validation metrics on results.
- **Palette_detection**: Program for palette detection.
- **Results**: Contains the data folder with the key materials for analysis and results from these analysis.

### Data Breakdown:

The `Results/Data` folder contains organized data and results for various materials and tests. Key subfolders include:

- **Clay**: Clay samples with varying concentrations.
- **Gips**: Gypsum samples with varying concentrations.
- **Milk**: Milk samples with varying concentrations.
- **Spinat**: Spinach samples with varying concentrations.
- **Water**: General water test results.
