"use client";

import { useState, useEffect, useRef } from "react";
import DashboardLayout from "@/components/dashboard-layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ScrollArea } from "@/components/ui/scroll-area";
import AdminDashboardLayout from "@/components/admin-dashboard-layout";

interface DocumentRecord {
  _id: string;
  name: string;
  upload_date: string;
  category?: string;
}

interface CategoryGuide {
  category: string;
  label: string;
  subcategories: Array<{
    name: string;
    description?: string;
  }>;
}

const categoryGuides: CategoryGuide[] = [
  {
    category: "program_details_documents",
    label: "Program Details",
    subcategories: [
      { name: "Course Query" },
      { name: "Attendance/Counselling Support" },
      { name: "Leave", description: "Leave policies" },
      {
        name: "Late Evaluation Submission",
        description: "Submission policies",
      },
      {
        name: "Missed Evaluation Submission",
        description: "Evaluation policies",
      },
      { name: "Withdrawal", description: "Withdrawal policies" },
    ],
  },
  {
    category: "curriculum_documents",
    label: "Curriculum Documents",
    subcategories: [
      { name: "Evaluation Score" },
      { name: "MAC", description: "Masai Additional Curriculum" },
      { name: "Revision", description: "Course content revision" },
      { name: "IA Support", description: "Technical support from IA" },
    ],
  },
  {
    category: "qa_documents",
    label: "FAQ",
    subcategories: [
      { name: "Product Support" },
      { name: "NBFC/ISA", description: "Financial FAQs" },
      { name: "Feedback" },
      { name: "Referral" },
      { name: "Personal Query" },
      { name: "Code Review" },
      { name: "Placement Support - Placements" },
      { name: "Offer Stage- Placements" },
      { name: "ISA/EMI/NBFC/Glide Related - Placements" },
      { name: "Session Support - Placement" },
    ],
  },
];

export default function AdminDocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function fetchDocuments() {
    setIsLoading(true);
    setError("");
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE}/v1/admin/documents`,
        { credentials: "include" }
      );
      if (!res.ok) throw new Error("Failed to fetch documents");
      const data = await res.json();
      setDocuments(Array.isArray(data) ? data : data.documents || []);
    } catch (err) {
      setError("Error loading documents");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    fetchDocuments();
  }, []);

  async function handleDelete(docId: string) {
    if (!window.confirm("Are you sure you want to delete this document?"))
      return;
    setIsLoading(true);
    setError("");
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE}/v1/admin/documents/${docId}`,
        {
          method: "DELETE",
          credentials: "include",
        }
      );
      if (!res.ok) throw new Error("Delete failed");
      fetchDocuments();
    } catch (err) {
      setError("Failed to delete document");
      setIsLoading(false);
    }
  }

  async function handleUpload(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!fileInputRef.current?.files?.[0] || !selectedCategory) {
      setError("Please select a category and upload a file");
      return;
    }
    setUploading(true);
    setError("");
    try {
      const file = fileInputRef.current.files[0];
      const formData = new FormData();
      formData.append("file", file);
      formData.append("category", selectedCategory);
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE}/v1/admin/documents/upload`,
        {
          method: "POST",
          body: formData,
          credentials: "include",
        }
      );
      if (!res.ok) throw new Error("Upload failed");
      fileInputRef.current.value = "";
      setSelectedCategory("");
      fetchDocuments();
    } catch {
      setError("Failed to upload document");
    } finally {
      setUploading(false);
    }
  }

  return (
    <AdminDashboardLayout>
      <div className="p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">
          Document Management
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Upload Document</CardTitle>
            </CardHeader>
            <CardContent>
              {error && <div className="text-red-600 mb-4">{error}</div>}
              <form onSubmit={handleUpload} className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Category</label>
                  <Select
                    value={selectedCategory}
                    onValueChange={setSelectedCategory}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select a category" />
                    </SelectTrigger>
                    <SelectContent>
                      {categoryGuides.map((guide) => (
                        <SelectItem key={guide.category} value={guide.category}>
                          {guide.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Document</label>
                  <input
                    type="file"
                    ref={fileInputRef}
                    required
                    className="w-full border rounded bg-white px-2 py-1"
                  />
                </div>
                <Button
                  type="submit"
                  className="w-full"
                  disabled={uploading || !selectedCategory}
                >
                  {uploading ? "Uploading..." : "Upload Document"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Category Guide</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[50vh]">
                <Accordion type="single" collapsible className="w-full">
                  {categoryGuides.map((guide) => (
                    <AccordionItem key={guide.category} value={guide.category}>
                      <AccordionTrigger className="text-lg font-semibold hover:no-underline">
                        {guide.label}
                      </AccordionTrigger>
                      <AccordionContent>
                        <div className="grid">
                          {guide.subcategories.map((sub, idx) => (
                            <div key={idx} className="p-2">
                              <div className="font-medium text-gray-900">
                                {sub.name}{" "}
                                {sub.description ? `(${sub.description})` : ""}
                              </div>
                            </div>
                          ))}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </ScrollArea>
            </CardContent>
          </Card>

          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>Uploaded Documents</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div>Loading...</div>
              ) : (
                <ul className="space-y-2">
                  {documents.length === 0 && (
                    <li className="text-gray-400">No documents found.</li>
                  )}
                  {documents.map((doc) => (
                    <li
                      key={doc._id}
                      className="flex items-center justify-between bg-gray-50 rounded p-3"
                    >
                      <div className="space-y-1">
                        <div className="font-medium">{doc.name}</div>
                        <div className="flex items-center space-x-2 text-sm text-gray-500">
                          <span>
                            {doc.upload_date
                              ? new Date(doc.upload_date).toLocaleString()
                              : ""}
                          </span>
                          {doc.category && (
                            <>
                              <span>•</span>
                              <span>
                                {categoryGuides.find(
                                  (g) => g.category === doc.category
                                )?.label || doc.category}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                      <Button
                        type="button"
                        size="sm"
                        variant="destructive"
                        onClick={() => handleDelete(doc._id)}
                        disabled={isLoading}
                      >
                        Delete
                      </Button>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </AdminDashboardLayout>
  );
}
