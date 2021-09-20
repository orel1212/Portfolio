using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace Project.Models
{
    public abstract class Person
    {
        [Key]
        [Required(ErrorMessage = "The ID is Required.")]
        [RegularExpression("^[0-9]{9}$", ErrorMessage = "The Id must contain 9digits")]
        public string ID { get; set; }

        [Required(ErrorMessage ="The Name is Required.")]
        [StringLength(10,ErrorMessage="The Name must be max 10chars long")]
        [RegularExpression("^[A-Z][a-z]+$", ErrorMessage = "The name is 1-10 length which must start with uppercase A-Z, and then just a-z lowercases.")]
        public string Name { get; set; }
        [Required(ErrorMessage = "The Password is Required.")]
        [StringLength(10)]
        [RegularExpression("^[A-Za-z0-9]+$", ErrorMessage = "The Password is 1-10 length which must contain only digits,uppercases A-Z and a-z lowercases")]
        public string Password { get; set; }
        [Required]
        public string Role { get; set; }
        public Person(string id, string name, string password, string Role)
        {
            this.ID = id;
            this.Name = name;
            this.Password = password;
            this.Role = Role;
        }

        public Person()
        {
            this.ID="";
            this.Name="";
            this.Password="";
            this.Role="";
        }
        public Person(Person p)
        {
            this.ID = p.ID;
            this.Name = p.Name;
            this.Password = p.Password;
            this.Role = p.Role;
        }
       
    }
}