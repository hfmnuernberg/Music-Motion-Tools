include("multiple_text_grab_context.js");
include("multiple_text_patcher.js");
include("multiple_text_callback_concat.js");
include("multiple_text_attributes.js");
include("multiple_text_TF.js");

autowatch = 1;
inlets = 10;

var g_inletsArray = [];
var g_glParams = [];
var g_text = null;

//var g_textObjsArray = [];
var g_textObj = null;
var g_fontsArray = [];

var g_tempWordTexcoordMatrix = new JitterMatrix(2, "float32", 1);
var g_tempLetterMat = new JitterMatrix(); // used in callback function
var g_tempCopyMasterTexcoord = new JitterMatrix(); // used in callback function

var g_inputMat = new JitterMatrix();
var g_matrixToMatchPlanes = new JitterMatrix(4, "float32", 1); //used to match planes between the input matrix and the param matrices
g_matrixToMatchPlanes.interp = 0;

// matrix used in the ScanWrapMatrix function, its purpose is explained there
var g_tempScanwrapMat = new JitterMatrix(1, "float32", 1); 
g_tempScanwrapMat.interp = 0;
g_tempScanwrapMat.adapt = 0;

var g_localInputParamMatrices = 
{
    positionMat:  new JitterMatrix(3, "float32", 1),
    colorMat:     new JitterMatrix(4, "float32", 1),
    scaleMat:     new JitterMatrix(3, "float32", 1),
    rotatexyzMat: new JitterMatrix(3, "float32", 1),
    rotateMat:    new JitterMatrix(4, "float32", 1),

    Init: function() 
    {   
        this.colorMat.interp = 0;
        this.colorMat.adapt = 0;
        this.scaleMat.interp = 0;
        this.scaleMat.adapt = 0;
        this.rotatexyzMat.interp = 0;
        this.rotatexyzMat.adapt = 0;
        this.rotateMat.interp = 0;
        this.rotateMat.adapt = 0;
    }, 

    SetMatricesDim: function(dim)
    {
        this.positionMat.dim = dim;
        this.colorMat.dim = dim;
        this.scaleMat.dim = dim;
        this.rotatexyzMat.dim = dim;
        this.rotateMat.dim = dim;
    },

    Destroy: function()
    {
        this.positionMat.freepeer();
        this.colorMat.freepeer();
        this.scaleMat.freepeer();
        this.rotatexyzMat.freepeer();
        this.rotateMat.freepeer();
    }

}

var g_compositeMatrixForTexture = new JitterMatrix(4, "float32", [1, 5]); // 5 rows, one for each glparam
g_compositeMatrixForTexture.setall(1,1,1,1);

var g_masterVerticesNormalsIndices = new JitterMatrix(8, "float32", [1,1]); // 4th plane of this matrix contains word ID
var g_masterVertMat =        new JitterMatrix(3, "float32", [1,1]);
var g_masterNormMat =        new JitterMatrix(3, "float32", [1,1]);
var g_masterVertIndicesMat = new JitterMatrix(4, "float32", [1,1]);
var g_masterTexCoordMat =    new JitterMatrix(2, "float32", [1,1]);
g_masterVertMat.adapt = 1;
g_masterNormMat.adapt = 1;
g_masterVertIndicesMat.adapt = 1;

var g_wordIndex = 0;

var g_concat = new JitterObject("jit.concat");
g_concat.mode = 0;
var g_unpack = new JitterObject("jit.unpack");
var g_scanwrap = new JitterObject("jit.scanwrap");
g_scanwrap.adapt = 0;
g_scanwrap.mode = 1;

function TextObj() 
{   
    this.glText = new JitterObject("jit.gl.text");
    this.glText.drawto = drawto;

    // this.glText.text("t");

    this.glText.color = [1,1,1,1];
    this.glText.mode = "3d";
    this.glText.depth = 0.2;
    this.glText.align = 1;

    // this.glText.position = [0,0,0];
    this.glText.automatic = 0;
    this.glText.matrixoutput = 2;

    this.listener = new JitterListener(this.glText.name, MatrixOutputCallback);

    this.SetAnchor = function(anchor)
    {
        this.glText.anchor = anchor;
    }

    this.SetAlign = function(align)
    {
        this.glText.align = align;
    }

    this.SetMaterial = function(matName)
    {
        this.glText.material = matName;
    }

    this.SetText = function(text)
    {   
        this.glText.text(text);
    }

    this.SetFontSize = function(fontSize)
    {
        this.glText.fontsize = fontSize;
    }

    this.SetFont = function(font) 
    {
        this.glText.fontname = (font);
    }

    this.SetDepth = function(depth) 
    {
        this.glText.depth = (depth);
    }

    this.SetScale = function(scale) 
    {
        this.glText.scale = scale;
    }

    this.SetMode = function(mode)
    {
        this.glText.mode = mode;
    }

    this.SetPosition = function(position)
    {
        this.glText.position = position;
    }

    this.SetColor = function(color)
    {
        this.glText.color = color;
    }

    this.SetWeight = function(weight)
    {
        this.glText.weight= weight;
    }

    this.Draw = function()
    {   
        this.glText.draw();
        // FF_Utils.Print("TEXT DRAW")
    }

    this.GetText = function()
    {
        return this.glText.text;
    }

    this.Destroy = function() 
    {
        this.glText.freepeer();
        // this.listener.freepeer();
    }
}

//-------------------------------
// PUBLIC FUNCTIONS 

function bang()
{
    g_mesh.draw();
}

function anything() // everything that doesn't match the attributes or public functions is sent to the mesh
{
    var args = arrayfromargs(arguments);
    if (typeof g_mesh[messagename] === "function") {
        g_mesh[messagename](args);
    } else {
        g_mesh[messagename] = args;
    }
}

function jit_matrix(matName)
{   
    if (enable) {
        g_inputMat.name = matName;

        if (g_glParams[inlet] == "position") {
            ScanWrapMatrix(g_inputMat, g_localInputParamMatrices.positionMat);
            AssignMatrixToCompositeMatrix(g_localInputParamMatrices.positionMat, "position")
            AssignMatrixToTexture();
        } 
        else if (g_glParams[inlet] == "scale") {
            ScanWrapMatrix(g_inputMat, g_localInputParamMatrices.scaleMat);
            AssignMatrixToCompositeMatrix(g_localInputParamMatrices.scaleMat, "scale");
            AssignMatrixToTexture();
        } 
        else if (g_glParams[inlet] == "color") {
            ScanWrapMatrix(g_inputMat, g_localInputParamMatrices.colorMat);
            AssignMatrixToCompositeMatrix(g_localInputParamMatrices.colorMat, "color");
            AssignMatrixToTexture(); 
        } 
        else if (g_glParams[inlet] == "rotatexyz") {
            ScanWrapMatrix(g_inputMat, g_localInputParamMatrices.rotatexyzMat);
            AssignMatrixToCompositeMatrix(g_localInputParamMatrices.rotatexyzMat, "rotatexyz");
            AssignMatrixToTexture();
        } 
        else if (g_glParams[inlet] == "rotate") {
            ScanWrapMatrix(g_inputMat, g_localInputParamMatrices.rotateMat);
            AssignMatrixToCompositeMatrix(g_localInputParamMatrices.rotateMat, "rotate");
            AssignMatrixToTexture();
        }
    }
}

function text() // CAUSES TO RECREATE TEXTS
{   
    var textArray = arrayfromargs(arguments);
    g_text = textArray;
    // FF_Utils.Print("TYOPE OF TEXT "+typeof(g_text) + " " +g_text)
    InitDrawAssign();
}

function clear()
{
    g_localInputParamMatrices.Init();

    g_text = null;
    DestroyTextObjs();
    InitMasterMatrices();
    AssignMatricesToBuffers();
    EnableTF(1);
}

function output_matrix()
{
    outlet(0, g_masterVertMat.name);
}

// ---------------------------------
// PRIVATE FUNCTIONS

function InitDrawAssign() // CAUSES TO RECREATE TEXTS
{   
    g_wordIndex = 0;
    InitMasterMatrices();
    SetTextAndDrawBang(); // here happens the concatenation of matrices
    UnpackMasterMatrixIntoSingleMatrices();
    AssignMatricesToBuffers();
    EnableTF(1);
}
InitDrawAssign.local = true;

function UnpackMasterMatrixIntoSingleMatrices()
{   
    // FF_Utils.Print("g_masterMat MAT "+g_masterVerticesNormalsIndices.name)

    g_unpack.jump = 3;
    g_unpack.offset = 0;
    g_masterVertMat.dim = g_masterVerticesNormalsIndices.dim;
    g_unpack.matrixcalc(g_masterVerticesNormalsIndices, g_masterVertMat);
    // FF_Utils.Print("g_masterVertMat MAT "+g_masterVertMat.name)

    g_unpack.jump = 3;
    g_unpack.offset = 5;
    g_masterNormMat.dim = g_masterVerticesNormalsIndices.dim;
    g_unpack.matrixcalc(g_masterVerticesNormalsIndices, g_masterNormMat);
    // FF_Utils.Print("g_masterNORMMat MAT "+g_masterNormMat.name)

    g_unpack.jump = 4;
    g_unpack.offset = 0;
    g_masterVertIndicesMat.dim = g_masterVerticesNormalsIndices.dim;
    g_unpack.matrixcalc(g_masterVerticesNormalsIndices, g_masterVertIndicesMat);
    // FF_Utils.Print("INDICES MAT NAME "+g_masterVertIndicesMat.name)
}
UnpackMasterMatrixIntoSingleMatrices.local = true;

function ScanWrapMatrix(maxMat, localMat)
{     
    if (Array.isArray(maxMat.dim) && maxMat.dim[1] != 1) {
        // copy input matrix to have always correct number of planes relative to local matrix
        g_matrixToMatchPlanes.planecount = localMat.planecount;
        g_matrixToMatchPlanes.dim = maxMat.dim;
        g_matrixToMatchPlanes.frommatrix(maxMat); 

        // adjust scanwrap according to local mat but use dimensions of input matrix
        g_scanwrap.dim = maxMat.dim[0]*maxMat.dim[1];
        g_scanwrap.planecount = localMat.planecount;
        g_scanwrap.type = localMat.type;

        // Use this matrix as a temporary output target for scanwrap, we do this because:
        // the input matrix could not have the same dimension as the localMat
        // so by assigning the output of scanwrap to this temp matrix, we can then copy the temp matrix into the localMat
        // which will then have the correct dimensions and will be filled with the cells of the input mat, even if this last one
        // had different dimensions than the localMat.
        g_tempScanwrapMat.planecount = localMat.planecount;
        g_tempScanwrapMat.dim = maxMat.dim[0]*maxMat.dim[1];

        g_scanwrap.matrixcalc(g_matrixToMatchPlanes.name, g_tempScanwrapMat.name);
        localMat.frommatrix(g_tempScanwrapMat);
    } else {
        localMat.frommatrix(maxMat);
        // FF_Utils.Print("MAT NAME 2: "+localMat.name)
    }
}
ScanWrapMatrix.local = true;

function SetTextAndDrawBang()
{   
    g_wordIndex = 0;
    if (Array.isArray(g_text)) {
        g_texture.SetDim(g_text.length);
        g_compositeMatrixForTexture.dim = [g_text.length, 5];
        g_compositeMatrixForTexture.setall(1,1,1,1);

        g_localInputParamMatrices.SetMatricesDim(g_text.length);

        var fontsCounter = 0;

        for (var i=0; i<g_text.length; i++) {
            DestroyTextObjs(); // necessary for single letters, otherwise jit.gl.text keeps the first letter received
            CreateTextObjs();
            if (Array.isArray(g_fontsArray)) {
                g_textObj.SetFont(g_fontsArray[fontsCounter]);
                fontsCounter++;
                fontsCounter %= g_text.length;
                fontsCounter %= g_fontsArray.length;
            } else {
                g_textObj.SetFont(g_fontsArray);
            }
            g_textObj.SetText(g_text[i]);
            g_textObj.Draw();
            g_wordIndex++;
        }
    } else if (g_text != null) {
        // FF_Utils.Print("TEXT "+g_text)
        g_texture.SetDim(1);
        g_compositeMatrixForTexture.dim = [1, 5];
        g_localInputParamMatrices.SetMatricesDim(1);
        DestroyTextObjs();
        CreateTextObjs();
        g_textObj.SetText(g_text);
        g_textObj.Draw();
        g_wordIndex++;
    }
}
SetTextAndDrawBang.local = true;

function InitMasterMatrices()
{       
    InitMasterMatrix(g_masterVerticesNormalsIndices, 8);
    InitMasterMatrix(g_masterVertMat, 3);
    InitMasterMatrix(g_masterNormMat, 3);
    InitMasterMatrix(g_masterVertIndicesMat, 4);
    InitMasterMatrix(g_masterTexCoordMat, 2);
}
InitMasterMatrices.local = true;

function InitMasterMatrix(matrix, planecount)
{   
    matrix.clear();
    matrix.planecount = planecount;
    matrix.dim = [1,1];
}
InitMasterMatrix.local = true;

function SetAllTextAttributes()
{   
    if (g_textObj != null) {
        g_textObj.SetMode(mode);
        // g_textObj.SetFont(font);
        g_textObj.SetDepth(depth)
        g_textObj.SetWeight(weight);
        g_textObj.SetAlign(align);
    }

} 
SetAllTextAttributes.local = true;

function CreateTextObjs()
{    
    if (g_textObj == null) {
        g_textObj = new TextObj();
        SetAllTextAttributes();
    }
}
CreateTextObjs.local = true;

//-------------------------------
// CLEAN UP
function DestroyTextObjs()
{   
    if (g_textObj != null) {
        g_textObj.Destroy();
        g_textObj = null;
    }
}
DestroyTextObjs.local = true;

function notifydeleted() 
{   
    DestroyTextObjs();

    EnableTF(0);

    g_texture.DestroyTextures();
    g_buffers.DestroyBuffers();
    g_TF.freepeer();
    g_mesh.freepeer();

    g_masterVerticesNormalsIndices.freepeer();
    g_masterVertMat.freepeer();
    g_masterNormMat.freepeer();
    g_masterVertIndicesMat.freepeer();

    g_compositeMatrixForTexture.freepeer();
    g_matrixToMatchPlanes.freepeer();
    g_tempScanwrapMat.freepeer();

    g_tempWordTexcoordMatrix.freepeer();
    g_tempLetterMat.freepeer();
    g_tempCopyMasterTexcoord.freepeer();
    g_inputMat.freepeer();
    g_masterTexCoordMat.freepeer();

    g_localInputParamMatrices.Destroy();

    g_concat.freepeer();
    g_unpack.freepeer();
    g_scanwrap.freepeer();
}
notifydeleted.local = true;