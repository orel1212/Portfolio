using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using Project.Models;
using System.Data.Entity;
namespace Project.DAL
{
    public class DataLayer : DbContext
    {
        public DbSet<Order> Orders { get; set; }
        public DbSet<OrderDescription> ODs { get; set; }
        public DbSet<Product> Products { get; set; }
        public DbSet<Person> Persons { get; set; }
        protected override void OnModelCreating(DbModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);
            modelBuilder.Entity<Order>().ToTable("Orders");
            modelBuilder.Entity<Person>().ToTable("Persons");
            modelBuilder.Entity<OrderDescription>().HasKey(od => new
            {
                od.OID,
                od.PID
            });
            modelBuilder.Entity<OrderDescription>().ToTable("OrderDescriptions");
            modelBuilder.Entity<Product>().ToTable("Products");
        }
    }
}