function getImagePath(imgDir, imgName, imgType) {
    return imgDir + "\\" + imgName + "_" + imgType + ".png";
}

function saveAsPng(fileName) {
    pngSaveOptions = new PNGSaveOptions()
    pngSaveOptions.compression = 9;
    fileRef = new File(fileName);
    app.activeDocument.saveAs(fileRef, pngSaveOptions, true, Extension.LOWERCASE);
}

function imgNameFromShadName(fileName) {
    return fileName.split("_shad.png")[0];
}

function imgNameFromFullPath(filePath) {
    parts = filePath.split("\\");
    return parts[parts.length-1]
}

// Set the ruler units to pixels
var originalRulerUnits = app.preferences.rulerUnits
app.preferences.rulerUnits = Units.PIXELS


var imgDir = "***img_folder***";
var imgName = "***img_name***";

var shadFileRef = new File(getImagePath(imgDir, imgName, "shad"));
var docRef = app.open(shadFileRef);

// turn it into a normal layer
var shadLayer = docRef.artLayers[0]
shadLayer.name = "image"
shadLayer.isBackgroundLayer = false;
shadLayer.kind = LayerKind.NORMAL;

// open mask image
var maskFileRef = new File(getImagePath(imgDir, imgName, "smask"));
app.open(maskFileRef);

// copy the image and close the doc
app.activeDocument.selection.selectAll();
app.activeDocument.selection.copy()
app.activeDocument.close(SaveOptions.DONOTSAVECHANGES)
// create a layer for it and paste
maskLayer = docRef.artLayers.add()
maskLayer.name = "mask"
app.activeDocument.selection.selectAll();
app.activeDocument. paste();

// select white region from the mask
doAction("selectHighlights", "customActions");

maskLayer.visible = false
docRef.activeLayer = shadLayer

doAction("contentAwareFill", "customActions");

saveAsPng(getImagePath(imgDir, imgName, "gunshadp")); 
docRef.close(SaveOptions.DONOTSAVECHANGES);

// restore unit settings
app.preferences.rulerUnits = originalRulerUnits
