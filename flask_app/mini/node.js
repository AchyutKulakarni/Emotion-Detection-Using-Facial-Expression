var mysql=require("mysql");
var express=require("express");
var con=mysql.createConnection({
    host:"127.0.0.1",
    user:"root",
    password:"",
    database:"emotion"

});
con.connect(function(err){
    if (err){
        console.log("Couldn't connect to the database");
        throw err;
    }
    console.log("Connected to the Database");

    con.query("CREATE DATABASE IF NOT EXISTS emotion",function(err,result){
        if (err){
            console.log("Database not create");
            throw err;
        }
        console.log("Database created");
        
    })

    con.query("CREATE TABLE IF NOT EXISTS reglog(Username varchar(30),Password varchar(10))",function(err,result){
        if(err){
            console.log(" Registration Table is not created");
            throw err;

        }
        console.log(" Registration Table created");
    });

    con.query("CREATE TABLE IF NOT EXISTS loginlog(Username varchar(30),Password varchar(10))",function(err,result){
        if(err){
            console.log("Login Table is not created");
            throw err;

        }
        console.log("Login Table created");
    });
});

var app=express()
var parser=express.urlencoded({extended:true});
var port=3000;

app.get('/',(req,res)=>{
    res.sendFile(__dirname+'/index.html');
});

app.post('/registerbtn',parser,(req,res)=>{
    var name=req.body.registerUsername
    var pass=req.body.registerPassword
    var conpass=req.body.confirmPassword

    if (!name || !pass|| !conpass) {
        console.log("Username or password is null. Data not inserted.");
        
    }

    else{
        if(pass!==conpass){
            console.log("Passwords are different");
            //return;
        }
        else{
            var checkUsernameQuery = "SELECT * FROM reglog WHERE Username = ?";
            con.query(checkUsernameQuery, [name], function (err, rows, fields) {
                if (err) {
                    console.log("Error checking username:", err);
                    throw err;
                }

                if (rows.length > 0) {
                    console.log("Username already taken. Data not inserted.");
                    return res.status(400).send("Username already taken. Please choose another username.");
                }
                var sql="INSERT INTO reglog(Username,Password) VALUES (?,?)"
                var values=[name,pass];

                con.query(sql,values,function(err,result){
                    if(err){
                        console.log("Data not inserted into reg table");
                        throw err;
                    }
                    console.log("data inserted into reg table");
                    res.sendFile(__dirname+'/index.html');

                });
            });
        }
        
    }
});



app.post('/loginbtn', parser, (req, res) => {
    var lname = req.body.loginUsername;
    var lpass = req.body.loginPassword;

    if (!lname || !lpass) {
        console.log("Username or password is null. Data not inserted.");
        return;
    } else {
        var check = 'SELECT * FROM reglog WHERE Username=?';
        con.query(check, [lname], function (err, rows) {
            if (err) {
                console.log("Error checking login username:", err);
                throw err;
            }
            if (rows.length > 0) {
                var checkpass = 'SELECT Password FROM reglog WHERE username=?';
                con.query(checkpass, [lname], function (err, result) {
                    if (err) {
                        console.log("Error checking login password", err);
                        throw err;
                    }

                    // Compare the entered password with the actual password from the database
                    if (lpass === result[0].Password) {
                        var sql = "INSERT INTO loginlog(Username, Password) VALUES (?, ?)";
                        var values = [lname, lpass];

                        con.query(sql, values, function (err, result) {
                            if (err) {
                                console.log("Data not inserted into log table");
                                throw err;
                            }
                            console.log("Data inserted into log table");
                            res.redirect('http://127.0.0.1:5000/');
                        });
                    } else {
                        res.send("Wrong Password");
                    }
                });
            } else {
                res.send("Username not found");
            }
        });
    }
});


var server=app.listen(9000,function(){
    console.log("server is running at http://localhost:9000");
});