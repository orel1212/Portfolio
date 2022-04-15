using Project.DAL;
using Project.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Project.ViewModels
{
    public class ProductsVM
    {
        public List<Product> existProducts { get; set; }
        public ProductsVM()
        {
            setProducts();
        }
        private void setProducts()
        {
            DataLayer dal = new DataLayer();
            this.existProducts = dal.Products.ToList<Product>();
        }
      
        
    }
}