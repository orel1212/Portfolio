using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace Project.Models
{
    public class User
    {
        [Key]
        [Required(ErrorMessage = "The ID is Required.")]
        [RegularExpression("^[0-9]{9}$", ErrorMessage = "The Id must contain 9digits")]
        public string ID { get; set; }

        [Required(ErrorMessage = "The Password is Required.")]
        [StringLength(10,ErrorMessage = "The Password is max 10chars long.")]
        [RegularExpression("^[A-Za-z0-9]+$", ErrorMessage = "The Password is 1-10 length which must contain only digits,uppercases A-Z and a-z lowercases")]
        public string Password { get; set; }
        public User(string id,string password)
        {
            this.ID = id;
            this.Password = password;
        }
        public User()
        {
            this.ID = "";
            this.Password = "";
        }
    }
}