document.addEventListener('DOMContentLoaded', function() {
    // Get the button element
    var scrollToFooterButton = document.getElementById('scrollToFooterButton');

    // Add a click event listener to the button
    scrollToFooterButton.addEventListener('click', function() {
        // Get the position of the footer element
        var footerPosition = document.getElementById('footer').getBoundingClientRect().top;

        // Scroll the page to the footer position
        window.scrollTo({
            top: footerPosition,
            behavior: 'smooth' // Use smooth scrolling for a better user experience
        });
    });
});
document.addEventListener('DOMContentLoaded',function(){
    var scrollregister=document.getElementById('scrollregister');

    scrollregister.addEventListener('click',function(){
        var regposition=document.getElementById('register').getBoundingClientRect().top;

        window.scrollTo({
            top:regposition,
            behavior:'smooth'
        })
    })
});

document.getElementById('showRegisterForm').addEventListener('click', function() {
    document.getElementById('loginForm').style.display = 'none';
    document.getElementById('registerForm').style.display = 'block';
});
  
document.getElementById('showLoginForm').addEventListener('click', function() {
    document.getElementById('loginForm').style.display = 'block';
    document.getElementById('registerForm').style.display = 'none';
});



// document.addEventListener('DOMContentLoaded', function() {
//     var registerForm = document.getElementById('registerForm');
//     var loginForm = document.getElementById('loginForm');
//     document.getElementById('showRegisterForm').addEventListener('click', function() {
//         registerForm.style.display = 'block';
//         loginForm.style.display = 'none';
//     });
//     document.getElementById('showLoginForm').addEventListener('click', function() {
//         registerForm.style.display = 'none';
//         loginForm.style.display = 'block';
//     });
//     document.getElementById('registerbtn').addEventListener('click', function() {
//         if (validateRegisterForm()) {
//             alert('Registration successful!');
//         }
//     });
//     document.getElementById('loginbtn').addEventListener('click', function() {
//         if (validateLoginForm()) {
//             alert('Login successful!');
//         }
//     });
//     function validateRegisterForm() {
//         var username = document.getElementById('registerUsername').value.trim();
//         var password = document.getElementById('registerPassword').value.trim();
//         var confirmPassword = document.getElementById('confirmPassword').value.trim();
//         if (username === '' || password === '' || confirmPassword === '') {
//             alert('Please fill in all fields');
//             return false;
//         }
//         if (password !== confirmPassword) {
//             alert('Passwords do not match');
//             return false;
//         }
//         return true;
//     }
//     function validateLoginForm() {
//         var username = document.getElementById('loginUsername').value.trim();
//         var password = document.getElementById('loginPassword').value.trim();
//         if (username === '' || password === '') {
//             alert('Please fill in all fields');
//             return false;
//         }
//         return true;
//     }
// });
function validater(){
    var reguser=document.getElementById("registerUsername").value.trim();
    var regpass=document.getElementById("registerPassword").value.trim();
    var regcon=document.getElementById("confirmPassword").value.trim();
    if (reguser === '' || regpass === '' || regcon=== '') {
        alert('Please fill in all fields');
        return false;
    }

    if(regpass !== regcon){
        alert("Your passwords did not match");
        return false;
    }

    return true;
}

function validate(){
    var loguser=document.getElementById("loginUsername").value.trim();
    var logpass=document.getElementById("loginPassword").value.trim();
    if (loguser === '' || logpass === '' ) {
        alert('Please fill in all fields');
        return false;
    }
}






