var g_texture =
{   
    // 0: scale, 1: rotaxyz, 2: newPos, 3: color, 4: rotate
    // this is now intended for the Y position of these submatrices in the textures
    texture: null,

    Init: function()
    {
        this.texture = new JitterObject("jit.gl.texture");
        this.texture.rectangle = 1;
        this.texture.type = "float32";
        this.texture.adapt = 0;
        this.texture.defaultimage = "white";
        this.texture.wrap  = "repeat repeat repeat";
        this.texture.filter  = "none";
    },

    FillTexture: function(matrix)
    {
        this.texture.jit_matrix(matrix.name);
    },

    SetDim: function(dim)
    {   
        // FF_Utils.Print("DIM "+dim)
        this.texture.dim = [dim, 5];
    },

    GetDim: function()
    {
        return this.texture.dim;
    },

    GetName: function()
    {   
        return this.texture.name;
    }, 

    SetDrawto: function(val)
    {
        this.texture.drawto = val;
    },

    Enable: function(val)
    {
       this.texture.enable = val;
    },

    DestroyTextures: function()
    {
        this.texture.freepeer();
    }
}
g_texture.Init();

var g_TFShader = new JitterObject("jit.gl.shader");
g_TFShader.file = "textmult.tf.jxs";

var g_buffers = 
{   
    positionLoop: new JitterObject("jit.gl.buffer.wrapper"),
    normals: new JitterObject("jit.gl.buffer.wrapper"),
    color: new JitterObject("jit.gl.buffer.wrapper"),

    origVertices_indices: new JitterObject("jit.gl.buffer.wrapper"), 
    origNormals: new JitterObject("jit.gl.buffer.wrapper"),

    Init: function()
    {
        this.positionLoop.type = "position";
        this.positionLoop.outname = "oVertices";
        this.normals.type = "normal";
        this.normals.outname = "oNormals";
        this.color.type = "color";
        this.color.outname = "oColor";

        this.origVertices_indices.type = "vertex_attr0";
        this.origNormals.type = "vertex_attr1";
    },

    GetNames: function() 
    {   
        return [ this.positionLoop.name, this.normals.name, this.color.name, this.origVertices_indices.name, this.origNormals.name ];
    },

    Enable: function(val)
    {
        this.positionLoop.enable = val;
        this.normals.enable = val;
        this.color.enable = val;
        this.origVertices_indices.enable = val;
        this.origNormals.enable = val;        
    },

    DestroyBuffers: function()
    {
        this.positionLoop.freepeer();
        this.normals.freepeer();
        this.color.freepeer();
        this.origVertices_indices.freepeer();
        this.origNormals.freepeer();        
    }
}
g_buffers.Init();

var g_TF = new JitterObject("jit.gl.tf");
g_TF.shader = g_TFShader.name;
g_TF.numbuffers = 5;
g_TF.bufnames = g_buffers.GetNames();
g_TF.texture = g_texture.GetName();

var g_mesh = new JitterObject("jit.gl.mesh");
g_mesh.draw_mode = "triangles";
g_mesh.color = [1, 1, 1, 1];
g_mesh.tf_name = g_TF.name;

//-------------------
// TF RELATED FUNCTIONS

function AssignMatricesToBuffers()
{   
    g_buffers.positionLoop.jit_matrix(g_masterVertMat.name);
    g_buffers.origVertices_indices.jit_matrix(g_masterVertIndicesMat.name);
    g_buffers.origNormals.jit_matrix(g_masterNormMat.name);
    g_buffers.normals.jit_matrix(g_masterNormMat.name);
    var whiteMatrix = new JitterMatrix(4, "float32", 1);
    whiteMatrix.dim = g_masterVertMat.dim;
    whiteMatrix.setall(1,1,1,1);
    g_buffers.color.jit_matrix(whiteMatrix.name);
    whiteMatrix.freepeer();
    g_mesh.texcoord_matrix(g_masterTexCoordMat.name);
    // g_mesh.color_matrix(g_inputColorMat.name);
}
AssignMatricesToBuffers.local = true;

function AssignMatrixToCompositeMatrix(matrix, type)
{
    g_compositeMatrixForTexture.usedstdim = 1;
    if (type == "position") {
        g_compositeMatrixForTexture.dstdimstart = [0,2];
        g_compositeMatrixForTexture.dstdimend = [matrix.dim-1, 2];
    } else if (type == "scale") {
        g_compositeMatrixForTexture.dstdimstart = [0,0];
        g_compositeMatrixForTexture.dstdimend = [matrix.dim-1, 0];
    } else if (type == "rotatexyz") {
        g_compositeMatrixForTexture.dstdimstart = [0,1];
        g_compositeMatrixForTexture.dstdimend = [matrix.dim-1, 1];
    } else if (type == "color") {
        g_compositeMatrixForTexture.dstdimstart = [0,3];
        g_compositeMatrixForTexture.dstdimend = [matrix.dim-1, 3];
    } else if (type == "rotate") {
        g_compositeMatrixForTexture.dstdimstart = [0,4];
        g_compositeMatrixForTexture.dstdimend = [matrix.dim-1, 4];
    }
    g_compositeMatrixForTexture.frommatrix(matrix.name);
}
AssignMatrixToCompositeMatrix.local = true;

function AssignMatrixToTexture() 
{   
    g_texture.FillTexture(g_compositeMatrixForTexture);
}
AssignMatrixToTexture.local = true;

function EnableTF(val)
{
    g_TF.enable = val;
}
EnableTF.local = true;

function SendMatDimToShader(matrix, type)
{   
    var paramName = "u_"+type+"_mat_dim";
    g_TFShader.param(paramName, matrix.dim);
}
SendMatDimToShader.local = true;