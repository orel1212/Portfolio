using Project.DAL;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;

namespace Project.Models
{
    public class Product
    {
        [Key]
        [Required]
        [DatabaseGenerated(DatabaseGeneratedOption.None)]
        public int PID { get; set; }

        [Required(ErrorMessage = "The Name is Required.")]
        [StringLength(10,ErrorMessage = "The Name is max 10chars long")]
        [RegularExpression("^[A-Z][a-z]+$", ErrorMessage = "The name is 1-10 length which must start with uppercase A-Z, and then just a-z lowercases(at least 1)")]
        public string Name { get; set; }
        [Required(ErrorMessage = "The Price is Required.")]
        [RegularExpression("^(^\\d{1,18}$)|(^\\d{1,18}\\.\\d{1,2}$)$",ErrorMessage="The price must look like be from 1 digit up to 18 before . and if there is . only up 2 digits after!")]
        public decimal Price { get; set; }
        [Required(ErrorMessage = "The Amount is Required.")]
        [RegularExpression("^[0-9]+$", ErrorMessage = "The amount must contain only digits(at least 1)")]
        public int Amount { get; set; }

        public Product(string name, decimal Price,int amount)
        {
            this.Name = name;
            this.Price = Price;
            this.PID = GetNextProductID();
            this.Amount = amount;
        }
        public Product()
        { 
        }
        private int GetNextProductID()
        {
            DataLayer dal = new DataLayer();
            List<Product> products = dal.Products.ToList<Product>();
            if (products.Count > 0)
                return products[products.Count - 1].PID + 1;
            return 100;
        }
    }
}