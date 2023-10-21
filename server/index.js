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

app.post('/render', (req, res) => {
    // Get the file that was set to our field named "image"
    //console.log(req);

    var cfg = req.body.cfg;
    var model = req.body.model;
    var clipskip = req.body.clipskip;
    var lora = req.body.lora;
    var prompt = req.body.prompt;

    var session = req.body.session;
    renderRecords[session] = 0;

    //console.log("session====" + session);

    var rawImg = req.files.imageByteArray.data
    //base64Data = rawImg.replace(/^data:image\/png;base64,/, ''),
    dirpath = __dirname + '/../capture/';
    imgname = uuidv4();
    imageFileName = imgname + '.png',
        imageLocation = dirpath + imageFileName;

    var batchcount = 4;
    var progress = 0;
    var currentImage = 0;
    var currentMask = 0;
    var totalMask = 0;

    var sampleSteps = 25;

    var buffer = Buffer.from(rawImg, "base64");

    fs.writeFile(imageLocation, buffer, { flag: "w" }, function (err) {
        if (err == null) {
            console.log("Start python");
            var dataToSend;
            // spawn new child process to call the python script
            //var prompt = "a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks.";

            var exec = require('child_process').exec;
            var exestring = 'python3.11 ../scripts/cli/main.py -n 1 -c ' + cfg +
                ' -i ' + imgname +
                ' -m ' + model +
                ' -cs ' + clipskip +
                ' -ss ' + sampleSteps +
                ' -l ' + lora +
                ' -b ' + batchcount +
                ' -p "' + prompt + '"';
            console.log(exestring);
            const python = exec(exestring);
            // collect data from script
            python.stdout.on('data', function (data) {
                console.log('Pipe data from python script ...' + data.toString());

                if (data.startsWith("controlnet_progress")) {
                    var p = parseInt(data.split(":")[1]);
                    progress = 50 * (currentImage / batchcount + (p / sampleSteps) / batchcount);
                    renderRecords[session] = progress;
                    //console.log("[progress 1]:" + progress);
                }
                else if (data.startsWith("controlnet_start")) {
                    currentImage = parseInt(data.split(":")[1]);
                    //console.log("currentImage 2 :" + currentImage);
                }
                else if (data.startsWith("inpaint_progress")) {
                    var p = parseInt(data.split(":")[1]);
                    progress = 50 + 50 * (currentImage / batchcount + (p / 11) / totalMask / batchcount + currentMask / totalMask / batchcount);
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
                console.error(`data: ${data}`);
            });

            // in close event we are sure that stream from child process is closed
            python.on('close', (code) => {
                console.log(`child process close all stdio with code ${code}`);
                // send data to browser
                res.json({
                    success: code == 0,
                    code: code,
                    data: imgname,
                });
            })
        }
        else {
            console.log("err" + err);
        }
    });

});

app.use('/output', express.static('../output'));

app.listen(port, () => {
    console.log(`Admin app listening on port ${port}`);
});