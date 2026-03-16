# Confirming Use of Google Patents Public Data for Benchmarks

This doc explains how to confirm that using **Google Patents Public Data** (BigQuery) for research and benchmark publication is allowed, and how to get written confirmation if you want it.

---

## 1. What the project uses

- **Source:** BigQuery public dataset `patents-public-data.patents.publications` (and related tables).
- **Provider:** Data is from IFI CLAIMS Patent Services and Google; see [Google Patents Public Datasets (BigQuery)](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data).

---

## 2. License already stated by Google (CC BY 4.0)

The official Google Cloud blog states that **“Google Patents Public Data” by IFI CLAIMS Patent Services and Google** is used under **CC BY 4.0** (Creative Commons Attribution 4.0):

- **Blog:** [Google Patents Public Datasets: connecting public, paid, and private patent data](https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data)  
- In the “Sources” lines under the example queries you’ll see:  
  **“Google Patents Public Data” by IFI CLAIMS Patent Services and Google, used under CC BY 4.0.**

Under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), you are allowed to:

- **Share** — copy and redistribute the material.
- **Adapt** — remix, transform, and build upon the material for **any purpose**, including commercial and **including creating benchmarks**.

You must give **appropriate credit** (attribution) and indicate if changes were made.

So: **creating and publishing a benchmark from this data is allowed under the license Google states**, as long as you provide proper attribution.

---

## 3. How you can confirm it yourself

### Step 1: Check the stated license (CC BY 4.0)

1. Open:  
   [Google Patents Public Datasets blog](https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data).
2. Scroll to any of the “Example queries” (e.g. “Peak year for patent filings per classification” or “Average number of inventors per patent, by country”).
3. Read the **“Source:”** line; it should say **“Google Patents Public Data” by IFI CLAIMS Patent Services and Google, used under CC BY 4.0**.

### Step 2: Check BigQuery public dataset docs

1. Open:  
   [BigQuery public datasets](https://docs.cloud.google.com/bigquery/public-data)  
   (search for “bigquery public data” on [cloud.google.com](https://cloud.google.com) if the URL changes).
2. Review the **Public Dataset Program** description (who pays for storage, who pays for queries, access methods).
3. There is no separate “benchmark use” section; use of the data is governed by the dataset’s license (here, CC BY 4.0 as stated in the blog) and Google Cloud’s general terms.

### Step 3: Check Google Cloud terms (optional)

1. Open:  
   [Google Cloud Terms of Service](https://cloud.google.com/terms)  
   and the **Service Specific Terms** (linked from there or at [cloud.google.com/terms/service-terms](https://cloud.google.com/terms/service-terms)).
2. “Customer Data” there refers to *your* data in GCP, not the contents of public datasets. Query results from public datasets are your data to use in line with the **dataset’s** license (CC BY 4.0 for Google Patents Public Data).
3. The **Benchmarking** section in the terms is about benchmarking *Google’s services*, not about using public dataset content for research benchmarks.

So your confirmation path is: **dataset license (CC BY 4.0) + normal use of your query results.**

### Step 4: Check the dataset in the Cloud Console (optional)

1. Go to:  
   [Google Cloud Console → Marketplace → Google Patents Public Datasets](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data).
2. See if the dataset page or “Overview” / “Documentation” links mention a license or terms; they often point back to the same program and the blog.

---

## 4. Getting written confirmation (optional)

If you want a written confirmation from Google that using this data for “research and benchmark publication” is allowed:

1. **Google Cloud Support**  
   - In the [Google Cloud Console](https://console.cloud.google.com), use **Support** (question mark or “Help”) and open a case.  
   - Ask: *“Is use of the BigQuery public dataset ‘Google Patents Public Data’ (patents-public-data) for creating and publishing research benchmarks permitted under the stated CC BY 4.0 license?”*

2. **Public datasets contact**  
   - For *listing* public datasets on Cloud Storage, Google gives: **gcp-public-data@google.com**.  
   - You can try the same address to ask whether the *use* of the **BigQuery** public dataset “Google Patents Public Data” for research/benchmark publication is permitted; they may redirect you to Support or the blog.

3. **Sales / enterprise**  
   - If you have an enterprise or sales contact, they can often get a formal answer from legal or product.

Keep the reply (e.g. screenshot or text) with your project records.

---

## 5. What to do when you publish the benchmark

- **Attribution:** In the benchmark’s README or paper, add something like:  
  *“Patent text in this benchmark is derived from **Google Patents Public Data** (IFI CLAIMS Patent Services and Google), used under **CC BY 4.0**.”*  
  Optionally link to the [blog](https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data) and/or the [BigQuery dataset](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data).
- **Your code/data license:** You can still license your own code and non-patent parts of the benchmark under any license you choose (e.g. MIT, Apache-2.0); the CC BY 4.0 obligation applies to the **patent data** (and derivatives of it), not necessarily to your entire repo.

---

## Summary

| Question | Answer |
|----------|--------|
| Does the stated license allow creating a benchmark? | **Yes.** Google states the data is used under **CC BY 4.0**, which allows adapt/derivative works (including benchmarks). |
| How do I confirm? | (1) Read the blog’s “Source: … CC BY 4.0” lines. (2) Check BigQuery public data docs. (3) Optionally ask Google Cloud Support or gcp-public-data@google.com for written confirmation. |
| What must I do when publishing? | Give **attribution** to “Google Patents Public Data” by IFI CLAIMS Patent Services and Google, used under CC BY 4.0. |
