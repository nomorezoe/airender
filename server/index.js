const express = require('express');
const fileUpload = require('express-fileupload');

const app = express();
const port = 3000;
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
//const { join } = require('path');
const cors = require('cors');
const socketIO = require('socket.io');
const bodyParser = require('body-parser');
var timeout = require('connect-timeout')

var renderRecords = {};

app.use(
    fileUpload({
        limits: {
            fileSize: 10000000,
        },
        abortOnLimit: true,
    })
);

app.use(bodyParser.urlencoded({
    extended: true
}));

app.use(cors({
    origin: '*'
}));

// Add this line to serve our index.html page
//app.use(express.static('public'));

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.get('/test_stress', (req, res) => {
    var exec = require('child_process').exec;
    var exestring = 'python3.11 ../scripts/cli/stress_test.py -n 1 -c 7 -i 5181d427-5400-4e17-808d-59e5b0b873e7 -m sd_xl_base -cs 1 -ss 30 -l None -p "drawing, style by NTY, 20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. highlighting their determined expressions. " -b 4 -cnm depth  -us 1 -st pencil -plc 1'

    
    const python = exec(exestring);
    // collect data from script
    python.stdout.on('data', function (data) {
        console.log(`data: ${data}`);
    });
    python.stderr.on('data', (data) => {
        console.error(`data: ${data}`);
    });

    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
    });
});

/*
app.get('/test_env2', (req, res) => {
    var exec = require('child_process').exec;
    var exestring = 'echo $PATH'

    const python = exec(exestring);
    // collect data from script
    python.stdout.on('data', function (data) {
        console.log(`data: ${data}`);
    });
    python.stderr.on('data', (data) => {
        console.error(`data: ${data}`);
    });

    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
    });
});

app.get('/test_env3', (req, res) => {
    var exec = require('child_process').exec;
    var exestring = 'python3 --version'

    const python = exec(exestring);
    // collect data from script
    python.stdout.on('data', function (data) {
        console.log(`data: ${data}`);
    });
    python.stderr.on('data', (data) => {
        console.error(`data: ${data}`);
    });

    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
    });
});

app.get('/test_env', (req, res) => {
    var exec = require('child_process').exec;
    var exestring = 'which python3'

    const python = exec(exestring);
    // collect data from script
    python.stdout.on('data', function (data) {
        console.log(`data: ${data}`);
    });
    python.stderr.on('data', (data) => {
        console.error(`data: ${data}`);
    });

    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
    });
});
*/
app.post('/render_progress', (req, res) => {
    //console.log(req);
    var session = req.body.session;
    //console.log("progress session====" + session);

    if (renderRecords[session] != undefined) {
        res.json({
            success: true,
            data: renderRecords[session],
        });
    }
    else {
        res.json({
            success: false,
            data: 0,
        });
    }
});

app.use('/render', function(req, res, next) {
    //req.clearTimeout(); // clear request timeout
    req.setTimeout(300000); //set a 20s timeout for this request
    next();
}).post('/render', (req, res) => {
    // Get the file that was set to our field named "image"
    //console.log(req);

    var cfg = req.body.cfg;
    var model = req.body.model;
    var clipskip = req.body.clipskip;
    var lora = req.body.lora;
    var prompt = req.body.prompt;

    var vae = (req.body.vae == 1)? 1:0;
    var sampleSteps = req.body.sampleSteps;
    var scheduler = req.body.scheduler;
    var inpaintStrength = req.body.inpaintStrength
    var controlnetModlel = req.body.controlnetModlel

    //add style
    var usestyle = 0;
    var style = "non_style";
    if(req.body.usestyle != undefined){
        usestyle = (req.body.usestyle == 1)?1:0;
    }
    if(req.body.style != undefined){
        style = req.body.style;
    }
    //


    var session = req.body.session;

    //console.log("session====" + session);

    var rawImg = req.files.imageByteArray.data;

    var rawImgGrid = req.files.gridImageByteArray.data;

    responseCallBack = function(value){
        res.json(value);
    }

    console.log(usestyle);
    console.log(style);
    
    generate(cfg, model, clipskip, lora, prompt, vae, sampleSteps, scheduler, inpaintStrength,
        controlnetModlel, session, rawImg, rawImgGrid, null, responseCallBack, usestyle, style);

        
});

app.use('/output', express.static('../output'));
app.use('/capture', express.static('../capture'));
app.use('/upscaled', express.static('../upscaled'));

app.listen(port, () => {
    console.log(`Admin app listening on port ${port}`);
});

//app.use(timeout(300000))


app.use('/upscale', function(req, res, next) {
    //req.clearTimeout(); // clear request timeout
    req.setTimeout(300000); //set a 20s timeout for this request
    next();
}).post('/upscale', (req, res) => {
    console.log("body"+ req.body);
    var filename = req.body.filename;

    var denoise = 0.5;
    if(req.body.denoise != undefined){
        denoise = parseFloat(req.body.denoise);
    }

    var exec = require('child_process').exec;
    var exestring = 'python3.11 ../scripts/cli/upscaler.py' + ' -i ' + filename 
                    +' -s 20 -d ' + denoise;

    console.log(exestring);
    const python = exec(exestring);

    responseCallBack = function(value){
        res.json(value);
    }
    // collect data from script
    python.stdout.on('data', function (data) {

    });
    python.stderr.on('data', (data) => {
        console.error(`data: ${data}`);
    });

    // in close event we are sure that stream from child process is closed
    python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        // send data to browser
        responseCallBack({
            success: code == 0,
            code: code,
            data: filename +"_upscale",
        });
    })

});

function generate(cfg, model, clipskip, lora, prompt, vae, sampleSteps, scheduler, inpaintStrength,
    controlnetModlel, session, rawImg, rawImgGrid, progressCallBack, responseCallBack, usestyle, style){


    
    renderRecords[session] = 0;

    //console.log("session====" + session);

   
    //base64Data = rawImg.replace(/^data:image\/png;base64,/, ''),
    var dirpath = __dirname + '/../capture/';
    var imgname = uuidv4();
    var imageFileName = imgname + '.png';
    var imageLocation = dirpath + imageFileName;

    var imageGridFileName = imgname + '_grid.png';
    var imageGridLocation = dirpath + imageGridFileName;

    var batchcount = 1;
    var progress = 0;

    var currentImage = 0;
    var currentMask = 0;
    var totalMask = 0;

    //var sampleSteps = 25;

    var buffer = Buffer.from(rawImg, "base64");
    var bufferGrid = Buffer.from(rawImgGrid, "base64");
    fs.writeFile(imageGridLocation, bufferGrid, { flag: "w" }, function (err) {
        if (err == null) {
            progress = 2
            renderRecords[session] = progress;
            fs.writeFile(imageLocation, buffer, { flag: "w" }, function (err) {
                progress = 4
                renderRecords[session] = progress;
                if (err == null) {
                    console.log("Start python");
                    var dataToSend;
                    // spawn new child process to call the python script
                    //var prompt = "a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks.";

                    
                    var exec = require('child_process').exec;
                    var exestring = 'python3.11 ../scripts/cli/main.py -n 1 -c ' + cfg +
                        ' -i ' + imgname +
                        ' -m deliberate_v4 ' + 
                        ' -cs ' + clipskip +
                        ' -ss ' + sampleSteps +
                        ' -l "' + lora + '"' + 
                        ' -b ' + batchcount +
                        ' -d 1 -ip 1' +
                        ' -v ' + vae +
                        ' -s ' + scheduler +
                        ' -is ' + inpaintStrength +
                        ' -cnm ' + controlnetModlel +
                        ' -us ' + '1' + 
                        ' --style ' + '"' + style + '"' + 
                        ' -p "' + prompt + '"';

                    console.log(exestring);
                    const python = exec(exestring);
                    // collect data from script
                    python.stdout.on('data', function (data) {
                        if(progress < 10){
                            progress = 10;
                            renderRecords[session] = progress;
                        }
                        console.log('Pipe data from python script ...' + data.toString());
                        if (data.startsWith("controlnet_progress")) {
                            var p = parseInt(data.split(":")[1]);
                            progress = 10 + 60 * (currentImage / batchcount + (p / sampleSteps) / batchcount);
                            renderRecords[session] = progress;
                            //console.log("[progress 1]:" + progress);
                        }
                        else if (data.startsWith("controlnet_start")) {
                            currentImage = parseInt(data.split(":")[1]);
                            //console.log("currentImage 2 :" + currentImage);
                        }
                        else if (data.startsWith("inpaint_progress")) {
                            var p = parseInt(data.split(":")[1]);
                            progress = 70 + 50 * (currentImage / batchcount + (p / 11) / totalMask / batchcount + currentMask / totalMask / batchcount);
                            //console.log("[progress 2]:" + progress);
                            renderRecords[session] = progress;
                        }
                        else if (data.startsWith("inpaint_start")) {
                            currentImage = parseInt(data.split(":")[1]);
                            //console.log("currentImage :" + currentImage);
                        }
                        else if (data.startsWith("inpaint_mask_start")) {
                            totalMask = parseInt(data.split(":")[2]);
                            currentMask = parseInt(data.split(":")[1]);
                            //console.log("totalMask" + totalMask + "currentMask" + currentMask);
                        }

                    });
                    python.stderr.on('data', (data) => {
                        console.error(`err data: ${data}`);
                    });

                    // in close event we are sure that stream from child process is closed
                    python.on('close', (code) => {
                        console.log(`child process close all stdio with code ${code}`);
                        // send data to browser
                        responseCallBack({
                            success: code == 0,
                            code: code,
                            data: imgname,
                        });
                    })
                }
                else {
                    console.log("err" + err);

                    responseCallBack({
                        success: false,
                        code: 100,
                        data: "image save failed",
                    });
                }
            });
        }
        else {
            console.log("err" + err);

            responseCallBack({
                success: false,
                code: 101,
                data: "grid image save failed",
            });
        }
    });
}

//socket
/*
const http = require('http');
const server = http.createServer(app);
const { Server } = require("socket.io");
const io = new Server(server);

io.on('connection', (socket) => {
    console.log('a user connected');
  
    socket.on("disconnect", (reason) => {
      console.log("a user left");
    })
  
    socket.on("generate", (text) => {
     
    })
  
  });
*/
