const express = require('express');
const fileUpload = require('express-fileupload');

const {spawn} = require('child_process');
const app = express();
const port = 3000;
const fs = require('fs');

app.use(
    fileUpload({
        limits: {
            fileSize: 10000000,
        },
        abortOnLimit: true,
    })
);

// Add this line to serve our index.html page
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.post('/upload', (req, res) => {
    // Get the file that was set to our field named "image"
    console.log(req.files);
    var rawImg = req.files.imageByteArray.data
    //base64Data = rawImg.replace(/^data:image\/png;base64,/, ''),
    dirpath = __dirname + '/test/',
    imageName = req.files.imageByteArray.name + '.png',
    imageLocation = dirpath + imageName;

    var buffer = Buffer.from(rawImg, "base64");

    fs.writeFile(imageLocation,buffer, { flag: "w"}, function(err) {
        if(err == null){
            var dataToSend;
            // spawn new child process to call the python script
            const python = spawn('python3', ['ControlNet-v1-1/process_img.py']);
            // collect data from script
            python.stdout.on('data', function (data) {
             console.log('Pipe data from python script ...'+ data.toString());
             dataToSend = data.toString();
             res.send(dataToSend);
            });
            // in close event we are sure that stream from child process is closed
            python.on('close', (code) => {
            console.log(`child process close all stdio with code ${code}`);
            // send data to browser
            res.send(dataToSend);
            })
        }
    });

});

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`);
});