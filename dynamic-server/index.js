const express = require("express");
const app = express();
const path = require('path');


var c = 'color:red;';
var fs = require("fs");
var text = fs.readFileSync("./mytext.txt");

app.set("view engine", "ejs");
app.use(express.static(path.join(__dirname, "public")));


app.get("/", (req, res) => {
    res.render("./index.ejs", { c, text , path: '/' });
   });


  

app.listen(3000, () => {
  console.log("server started on port 3000");
});

setInterval(function A() {
  console.log(text)
  app.get("/", (req, res) => {
    res.render("./index.ejs", { c, text , path: '/' });
   });
  text= fs.readFileSync("./mytext.txt");
}, 5000);